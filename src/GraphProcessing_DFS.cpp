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
// Public DFS method
std::vector<int> Graph::dfs(int startVertex, ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::Sequential:
            return dfsSequential(startVertex);
        case ExecutionMode::ParallelCPU:
            return dfsParallelCPU(startVertex);
        case ExecutionMode::ParallelGPU:
            return dfsParallelGPU(startVertex);
        default:
            throw std::invalid_argument("Invalid ExecutionMode.");
    }
}


// Private method for sequential DFS
std::vector<int> Graph::dfsSequential(int startVertex) {
    std::vector<bool> visited(numVertices, false);
    std::vector<int> visitOrder;
    std::stack<int> s;

    s.push(startVertex);

    while (!s.empty()) {
        int current = s.top();
        s.pop();

        if (!visited[current]) {
            visited[current] = true;
            visitOrder.push_back(current);

            // Add neighbors to the stack
            const auto& neighbors = adjacencyList[current];
            for (auto it = neighbors.rbegin(); it != neighbors.rend(); ++it) {
                int neighborVertex = it->first;
                if (!visited[neighborVertex]) {
                    s.push(neighborVertex);
                }
            }
        }
    }

    return visitOrder;
}

// Private method for parallel DFS using std::thread
std::vector<int> Graph::dfsParallelCPU(int startVertex) {
    std::vector<std::atomic<bool>> visited(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        visited[i] = false;
    }

    std::vector<int> visitOrder;
    std::mutex visitOrderMutex;

    std::stack<int> s;
    s.push(startVertex);

    while (!s.empty()) {
        int current = s.top();
        s.pop();

        // Atomically check and set the visited status
        bool expected = false;
        if (visited[current].compare_exchange_strong(expected, true)) {
            // Add current to visit order
            {
                std::lock_guard<std::mutex> lock(visitOrderMutex);
                visitOrder.push_back(current);
            }

            // Collect unvisited neighbors
            std::vector<int> unvisitedNeighbors;
            for (const auto& neighbor : adjacencyList[current]) {
                int neighborVertex = neighbor.first;
                if (!visited[neighborVertex]) {
                    unvisitedNeighbors.push_back(neighborVertex);
                }
            }

            // Process neighbors in parallel
            std::vector<std::thread> threads;
            std::mutex stackMutex;
            for (int neighborVertex : unvisitedNeighbors) {
                threads.emplace_back([&, neighborVertex]() {
                    // Atomically check and set the visited status
                    bool expectedNeighbor = false;
                    if (visited[neighborVertex].compare_exchange_strong(expectedNeighbor, true)) {
                        {
                            std::lock_guard<std::mutex> lock(visitOrderMutex);
                            visitOrder.push_back(neighborVertex);
                        }

                        // Recursively process neighbor's neighbors
                        processNeighborDFS(neighborVertex, visited, visitOrder, visitOrderMutex);
                    }
                });
            }

            // Join all threads
            for (auto& thread : threads) {
                thread.join();
            }
        }
    }

    return visitOrder;
}

// Helper private method for recursive DFS in parallel
void Graph::processNeighborDFS(
    int vertex,
    std::vector<std::atomic<bool>>& visited,
    std::vector<int>& visitOrder,
    std::mutex& visitOrderMutex
) {
    // Collect unvisited neighbors
    std::vector<int> unvisitedNeighbors;
    for (const auto& neighbor : adjacencyList[vertex]) {
        int neighborVertex = neighbor.first;
        if (!visited[neighborVertex]) {
            unvisitedNeighbors.push_back(neighborVertex);
        }
    }

    // Process neighbors in parallel
    std::vector<std::thread> threads;
    for (int neighborVertex : unvisitedNeighbors) {
        threads.emplace_back([&, neighborVertex]() {
            // Atomically check and set the visited status
            bool expectedNeighbor = false;
            if (visited[neighborVertex].compare_exchange_strong(expectedNeighbor, true)) {
                {
                    std::lock_guard<std::mutex> lock(visitOrderMutex);
                    visitOrder.push_back(neighborVertex);
                }

                // Recursively process neighbor's neighbors
                processNeighborDFS(neighborVertex, visited, visitOrder, visitOrderMutex);
            }
        });
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }
}


// Private method for parallel DFS using CUDA
std::vector<int> Graph::dfsParallelGPU(int startVertex) {
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
    int *d_visited = nullptr;
    int *d_stack = nullptr;
    int *d_stackTop = nullptr;
    int *d_visitOrder = nullptr;

    // Allocate device memory
    cudaMalloc(&d_edgeArray, edgeArray.size() * sizeof(int));
    cudaMalloc(&d_edgeOffsets, edgeOffsets.size() * sizeof(int));
    cudaMalloc(&d_visited, numVertices * sizeof(int));
    cudaMalloc(&d_stack, numVertices * sizeof(int));
    cudaMalloc(&d_stackTop, sizeof(int));
    cudaMalloc(&d_visitOrder, numVertices * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_edgeArray, edgeArray.data(), edgeArray.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeOffsets, edgeOffsets.data(), edgeOffsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize visited array and visitOrder on device
    cudaMemset(d_visited, 0, numVertices * sizeof(int));
    cudaMemset(d_visitOrder, -1, numVertices * sizeof(int));

    // Initialize stack with startVertex
    cudaMemcpy(&d_stack[0], &startVertex, sizeof(int), cudaMemcpyHostToDevice);
    int h_stackTop = 1;
    cudaMemcpy(d_stackTop, &h_stackTop, sizeof(int), cudaMemcpyHostToDevice);

    // Launch DFS kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;  // Start with one block

    dfsKernel<<<blocksPerGrid, threadsPerBlock>>>(
        numVertices,
        d_edgeArray,
        d_edgeOffsets,
        d_visited,
        d_stack,
        d_stackTop,
        d_visitOrder
    );

    cudaDeviceSynchronize();

    // Copy visitOrder from device to host
    std::vector<int> h_visitOrder(numVertices);
    cudaMemcpy(h_visitOrder.data(), d_visitOrder, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_edgeArray);
    cudaFree(d_edgeOffsets);
    cudaFree(d_visited);
    cudaFree(d_stack);
    cudaFree(d_stackTop);
    cudaFree(d_visitOrder);

    // Extract visited vertices in order
    std::vector<int> visitOrder;
    for (int i = 0; i < numVertices; ++i) {
        if (h_visitOrder[i] >= 0) {
            visitOrder.push_back(h_visitOrder[i]);
        }
    }

    return visitOrder;
}

// Device DFS kernel
__global__
void dfsKernel(
    int numVertices,
    const int *edgeArray,
    const int *edgeOffsets,
    int *visited,
    int *stack,
    int *stackTop,
    int *visitOrder
) {
    extern __shared__ int sharedStack[];

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / warpSize;
    int laneId = threadId % warpSize;

    // Each warp works together
    if (warpId == 0) {
        // Initialize shared stack
        int localStackTop = 0;
        if (laneId == 0) {
            localStackTop = atomicAdd(stackTop, 0);
        }
        localStackTop = __shfl_sync(0xFFFFFFFF, localStackTop, 0);

        while (localStackTop > 0) {
            int currentVertex;
            if (laneId == 0) {
                localStackTop--;
                currentVertex = stack[localStackTop];
            }
            currentVertex = __shfl_sync(0xFFFFFFFF, currentVertex, 0);

            // Atomically check and set the visited status
            int wasVisited = atomicExch(&visited[currentVertex], 1);
            if (wasVisited == 0) {
                // Mark visit order
                visitOrder[currentVertex] = currentVertex;

                int edgeStart = edgeOffsets[currentVertex];
                int edgeEnd = edgeOffsets[currentVertex + 1];
                int numNeighbors = edgeEnd - edgeStart;

                // Load neighbors into shared memory for coalesced access
                for (int i = laneId; i < numNeighbors; i += warpSize) {
                    int neighbor = edgeArray[edgeStart + i];

                    // Atomically check and set the visited status
                    int neighborVisited = atomicExch(&visited[neighbor], 1);
                    if (neighborVisited == 0) {
                        int pos;
                        if (laneId == 0) {
                            pos = atomicAdd(&localStackTop, 1);
                        }
                        pos = __shfl_sync(0xFFFFFFFF, pos, 0);
                        sharedStack[pos] = neighbor;
                    }
                }
            }

            // Sync threads before updating the global stack
            __syncwarp();

            // Copy shared stack back to global stack
            if (laneId == 0) {
                int globalTop = atomicAdd(stackTop, localStackTop);
            }
            __syncwarp();

            for (int i = laneId; i < localStackTop; i += warpSize) {
                int idx = atomicAdd(stackTop, -1);
                stack[idx - 1] = sharedStack[i];
            }

            __syncwarp();

            if (laneId == 0) {
                localStackTop = 0;
            }
        }
    }
}

} // namespace GraphProcessing
