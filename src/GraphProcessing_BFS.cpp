// GraphProcessing.cpp

#include "GraphProcessing.h"
#include <vector>
#include <limits>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <atomic>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace GraphProcessing {

// Performs Breadth-First Search (BFS) on the graph
    std::vector<int> Graph::bfs(int startVertex, ExecutionMode mode) {
        switch (mode) {
            case ExecutionMode::Sequential:
                return bfsSequential(startVertex);
            case ExecutionMode::ParallelCPU:
                return bfsParallelCPU(startVertex);
            case ExecutionMode::ParallelGPU:
                return bfsParallelGPU(startVertex);
            default:
                throw std::invalid_argument("Invalid ExecutionMode.");
        }
    }

    // Private method for sequential BFS
    std::vector<int> Graph::bfsSequential(int startVertex) {
        std::vector<bool> visited(numVertices, false);
        std::vector<int> visitOrder;
        std::queue<int> q;

        visited[startVertex] = true;
        q.push(startVertex);

        while (!q.empty()) {
            int current = q.front();
            q.pop();
            visitOrder.push_back(current);

            for (const auto& neighbor : adjacencyList[current]) {
                int neighborVertex = neighbor.first;
                if (!visited[neighborVertex]) {
                    visited[neighborVertex] = true;
                    q.push(neighborVertex);
                }
            }
        }

        return visitOrder;
    }

    // Private method for parallel BFS using std::thread
    std::vector<int> Graph::bfsParallelCPU(int startVertex) {
        std::vector<std::atomic<bool>> visited(numVertices);
        for (int i = 0; i < numVertices; ++i) {
            visited[i] = false;
        }

        std::vector<int> visitOrder;
        std::mutex visitOrderMutex;

        std::vector<int> frontier;
        frontier.push_back(startVertex);
        visited[startVertex] = true;

        while (!frontier.empty()) {
            std::vector<int> nextFrontier;
            std::mutex nextFrontierMutex;

            std::vector<std::thread> threads;
            size_t numThreads = std::thread::hardware_concurrency();
            size_t chunkSize = (frontier.size() + numThreads - 1) / numThreads;

            for (size_t t = 0; t < numThreads; ++t) {
                size_t start = t * chunkSize;
                size_t end = std::min(start + chunkSize, frontier.size());

                threads.emplace_back([&, start, end]() {
                    std::vector<int> localNextFrontier;
                    std::vector<int> localVisitOrder;

                    for (size_t i = start; i < end; ++i) {
                        int current = frontier[i];
                        localVisitOrder.push_back(current);

                        for (const auto& neighbor : adjacencyList[current]) {
                            int neighborVertex = neighbor.first;

                            // Atomically check and set the visited status
                            bool expected = false;
                            if (visited[neighborVertex].compare_exchange_strong(expected, true)) {
                                localNextFrontier.push_back(neighborVertex);
                            }
                        }
                    }

                    // Merge local results into shared data structures
                    {
                        std::lock_guard<std::mutex> lock(nextFrontierMutex);
                        nextFrontier.insert(nextFrontier.end(), localNextFrontier.begin(), localNextFrontier.end());
                    }
                    {
                        std::lock_guard<std::mutex> lock(visitOrderMutex);
                        visitOrder.insert(visitOrder.end(), localVisitOrder.begin(), localVisitOrder.end());
                    }
                });
            }

            // Wait for all threads to finish
            for (auto& thread : threads) {
                thread.join();
            }

            frontier = std::move(nextFrontier);
        }

        return visitOrder;
    }

    // Private method for parallel BFS using CUDA
std::vector<int> Graph::bfsParallelGPU(int startVertex) {
    // Ensure the startVertex is valid
    if (startVertex < 0 || startVertex >= numVertices) {
        throw std::out_of_range("Invalid start vertex.");
    }

    // Prepare the graph data in CSR format for efficient GPU processing
    std::vector<int> edgeArray;      // Flattened edge list
    std::vector<int> edgeOffsets;    // Index offsets for each vertex
    edgeOffsets.resize(numVertices + 1);

    int totalEdges = 0;
    for (int i = 0; i < numVertices; ++i) {
        edgeOffsets[i] = totalEdges;
        const auto& neighbors = adjacencyList[i];
        totalEdges += neighbors.size();
        for (const auto& neighbor : neighbors) {
            edgeArray.push_back(neighbor.first);
        }
    }
    edgeOffsets[numVertices] = totalEdges;

    // Device variables
    int *d_edgeArray = nullptr;
    int *d_edgeOffsets = nullptr;
    int *d_frontier = nullptr;
    int *d_nextFrontier = nullptr;
    int *d_visited = nullptr;
    int *d_frontierSize = nullptr;
    int *d_nextFrontierSize = nullptr;
    int *d_visitOrder = nullptr;

    // Allocate device memory
    cudaMalloc(&d_edgeArray, edgeArray.size() * sizeof(int));
    cudaMalloc(&d_edgeOffsets, edgeOffsets.size() * sizeof(int));
    cudaMalloc(&d_frontier, numVertices * sizeof(int));
    cudaMalloc(&d_nextFrontier, numVertices * sizeof(int));
    cudaMalloc(&d_visited, numVertices * sizeof(int));
    cudaMalloc(&d_frontierSize, sizeof(int));
    cudaMalloc(&d_nextFrontierSize, sizeof(int));
    cudaMalloc(&d_visitOrder, numVertices * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_edgeArray, edgeArray.data(), edgeArray.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeOffsets, edgeOffsets.data(), edgeOffsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize visited array and visitOrder on device
    cudaMemset(d_visited, 0, numVertices * sizeof(int));
    cudaMemset(d_visitOrder, -1, numVertices * sizeof(int));

    // Initialize frontier with startVertex
    int h_frontierSize = 1;
    cudaMemcpy(d_frontier, &startVertex, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontierSize, &h_frontierSize, sizeof(int), cudaMemcpyHostToDevice);

    // Mark startVertex as visited
    int one = 1;
    cudaMemcpy(&d_visited[startVertex], &one, sizeof(int), cudaMemcpyHostToDevice);

    int level = 0;

    // BFS loop
    while (h_frontierSize > 0) {
        // Reset next frontier size
        int zero = 0;
        cudaMemcpy(d_nextFrontierSize, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // Determine grid and block sizes
        int threadsPerBlock = 256;
        int blocksPerGrid = (h_frontierSize + threadsPerBlock - 1) / threadsPerBlock;

        // Launch BFS kernel
        bfsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            numVertices,
            d_edgeArray,
            d_edgeOffsets,
            d_frontier,
            d_frontierSize,
            d_nextFrontier,
            d_nextFrontierSize,
            d_visited,
            d_visitOrder,
            level
        );

        cudaDeviceSynchronize();

        // Prepare for next iteration
        cudaMemcpy(&h_frontierSize, d_nextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost);

        // Swap frontiers
        std::swap(d_frontier, d_nextFrontier);
        std::swap(d_frontierSize, d_nextFrontierSize);

        level++;
    }

    // Copy visitOrder from device to host
    std::vector<int> h_visitOrder(numVertices);
    cudaMemcpy(h_visitOrder.data(), d_visitOrder, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_edgeArray);
    cudaFree(d_edgeOffsets);
    cudaFree(d_frontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_visited);
    cudaFree(d_frontierSize);
    cudaFree(d_nextFrontierSize);
    cudaFree(d_visitOrder);

    // Extract visited vertices in order
    std::vector<int> visitOrder;
    for (int i = 0; i < numVertices; ++i) {
        if (h_visitOrder[i] >= 0) {
            visitOrder.push_back(i);
        }
    }

    return visitOrder;
}

// Device BFS kernel
__global__
void bfsKernel(
    int numVertices,
    const int *edgeArray,
    const int *edgeOffsets,
    const int *frontier,
    const int *frontierSize,
    int *nextFrontier,
    int *nextFrontierSize,
    int *visited,
    int *visitOrder,
    int level
) {
    // Shared memory for coalesced access
    __shared__ int s_frontier[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load frontier into shared memory
    if (idx < *frontierSize) {
        s_frontier[threadIdx.x] = frontier[idx];
    }
    __syncthreads();

    if (idx >= *frontierSize) return;

    int currentVertex = s_frontier[threadIdx.x];

    // Mark visit order (only once)
    visitOrder[currentVertex] = level;

    int edgeStart = edgeOffsets[currentVertex];
    int edgeEnd = edgeOffsets[currentVertex + 1];

    for (int i = edgeStart; i < edgeEnd; ++i) {
        int neighbor = edgeArray[i];

        // Check if neighbor has been visited
        if (atomicExch(&visited[neighbor], 1) == 0) {
            // Add neighbor to next frontier
            int pos = atomicAdd(nextFrontierSize, 1);
            nextFrontier[pos] = neighbor;
        }
    }
}

} // namespace GraphProcessing
