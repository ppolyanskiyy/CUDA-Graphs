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

// Public Dijkstra's algorithm method
std::vector<int> Graph::dijkstra(int startVertex, ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::Sequential:
            return dijkstraSequential(startVertex);
        case ExecutionMode::ParallelCPU:
            return dijkstraParallelCPU(startVertex);
        case ExecutionMode::ParallelGPU:
            return dijkstraParallelGPU(startVertex);
        default:
            throw std::invalid_argument("Invalid ExecutionMode.");
    }
}

// Private method for sequential Dijkstra's algorithm
std::vector<int> Graph::dijkstraSequential(int startVertex) {
    const int INF = std::numeric_limits<int>::max();
    std::vector<int> distances(numVertices, INF);
    distances[startVertex] = 0;

    // Min-heap priority queue: (distance, vertex)
    using pii = std::pair<int, int>;
    std::priority_queue<pii, std::vector<pii>, std::greater<pii>> minHeap;
    minHeap.emplace(0, startVertex);

    while (!minHeap.empty()) {
        int currentDist = minHeap.top().first;
        int currentVertex = minHeap.top().second;
        minHeap.pop();

        // Skip if we have already found a better path
        if (currentDist > distances[currentVertex]) {
            continue;
        }

        // Explore neighbors
        for (const auto& neighbor : adjacencyList[currentVertex]) {
            int neighborVertex = neighbor.first;
            int weight = neighbor.second;
            int newDist = distances[currentVertex] + weight;

            if (newDist < distances[neighborVertex]) {
                distances[neighborVertex] = newDist;
                minHeap.emplace(newDist, neighborVertex);
            }
        }
    }

    return distances;
}

// Private method for parallel Dijkstra's algorithm using std::thread
std::vector<int> Graph::dijkstraParallelCPU(int startVertex) {
    const int INF = std::numeric_limits<int>::max();
    std::vector<std::atomic<int>> distances(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        distances[i] = INF;
    }
    distances[startVertex] = 0;

    // Shared priority queue protected by mutex
    using pii = std::pair<int, int>;
    auto cmp = [](const pii& left, const pii& right) {
        return left.first > right.first;
    };
    std::priority_queue<pii, std::vector<pii>, decltype(cmp)> minHeap(cmp);
    minHeap.emplace(0, startVertex);
    std::mutex minHeapMutex;

    std::vector<std::thread> threads;
    size_t numThreads = std::thread::hardware_concurrency();

    std::atomic<bool> workDone(false);

    auto worker = [&]() {
        while (!workDone.load()) {
            pii current;
            {
                std::lock_guard<std::mutex> lock(minHeapMutex);
                if (minHeap.empty()) {
                    workDone.store(true);
                    return;
                }
                current = minHeap.top();
                minHeap.pop();
            }

            int currentDist = current.first;
            int currentVertex = current.second;

            // Skip if we have already found a better path
            if (currentDist > distances[currentVertex].load()) {
                continue;
            }

            // Explore neighbors
            for (const auto& neighbor : adjacencyList[currentVertex]) {
                int neighborVertex = neighbor.first;
                int weight = neighbor.second;
                int newDist = currentDist + weight;

                int oldDist = distances[neighborVertex].load();
                while (newDist < oldDist) {
                    if (distances[neighborVertex].compare_exchange_weak(oldDist, newDist)) {
                        std::lock_guard<std::mutex> lock(minHeapMutex);
                        minHeap.emplace(newDist, neighborVertex);
                        break;
                    }
                }
            }
        }
    };

    // Launch worker threads
    for (size_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker);
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Convert atomic distances to integer vector
    std::vector<int> result(numVertices);
    for (int i = 0; i < numVertices; ++i) {
        result[i] = distances[i].load();
    }

    return result;
}

// Private method for Dijkstra's algorithm using CUDA
std::vector<int> Graph::dijkstraParallelGPU(int startVertex) {
    // Ensure the startVertex is valid
    if (startVertex < 0 || startVertex >= numVertices) {
        throw std::out_of_range("Invalid start vertex.");
    }

    // Prepare the graph data in CSR format for efficient GPU processing
    std::vector<int> edgeArray;      // Flattened edge list
    std::vector<int> edgeWeights;    // Corresponding edge weights
    std::vector<int> edgeOffsets;    // Index offsets for each vertex
    edgeOffsets.resize(numVertices + 1);

    int totalEdges = 0;
    for (int i = 0; i < numVertices; ++i) {
        edgeOffsets[i] = totalEdges;
        const auto& neighbors = adjacencyList[i];
        totalEdges += neighbors.size();
        for (const auto& neighbor : neighbors) {
            edgeArray.push_back(neighbor.first);
            edgeWeights.push_back(neighbor.second);
        }
    }
    edgeOffsets[numVertices] = totalEdges;

    // Device variables
    int *d_edgeArray = nullptr;
    int *d_edgeWeights = nullptr;
    int *d_edgeOffsets = nullptr;
    int *d_distances = nullptr;
    int *d_updated = nullptr;
    int *d_over = nullptr;

    // Allocate device memory
    cudaMalloc(&d_edgeArray, edgeArray.size() * sizeof(int));
    cudaMalloc(&d_edgeWeights, edgeWeights.size() * sizeof(int));
    cudaMalloc(&d_edgeOffsets, edgeOffsets.size() * sizeof(int));
    cudaMalloc(&d_distances, numVertices * sizeof(int));
    cudaMalloc(&d_updated, numVertices * sizeof(int));
    cudaMalloc(&d_over, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_edgeArray, edgeArray.data(), edgeArray.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeWeights, edgeWeights.data(), edgeWeights.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeOffsets, edgeOffsets.data(), edgeOffsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize distances on device
    const int INF = INT_MAX;
    std::vector<int> h_distances(numVertices, INF);
    h_distances[startVertex] = 0;
    cudaMemcpy(d_distances, h_distances.data(), numVertices * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize updated array to zero
    cudaMemset(d_updated, 0, numVertices * sizeof(int));

    int h_over = 1;

    // Kernel execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (numVertices + threadsPerBlock - 1) / threadsPerBlock;

    while (h_over) {
        h_over = 0;
        cudaMemcpy(d_over, &h_over, sizeof(int), cudaMemcpyHostToDevice);

        // Launch Dijkstra kernel
        dijkstraKernel<<<blocksPerGrid, threadsPerBlock>>>(
            numVertices,
            d_edgeArray,
            d_edgeWeights,
            d_edgeOffsets,
            d_distances,
            d_updated,
            d_over
        );

        cudaDeviceSynchronize();

        // Copy 'over' flag from device to host
        cudaMemcpy(&h_over, d_over, sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Copy distances from device to host
    cudaMemcpy(h_distances.data(), d_distances, numVertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_edgeArray);
    cudaFree(d_edgeWeights);
    cudaFree(d_edgeOffsets);
    cudaFree(d_distances);
    cudaFree(d_updated);
    cudaFree(d_over);

    return h_distances;
}

// Device Dijkstra kernel
__global__
void dijkstraKernel(
    int numVertices,
    const int *edgeArray,
    const int *edgeWeights,
    const int *edgeOffsets,
    int *distances,
    int *updated,
    int *over
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertices) {
        int currentDist = distances[tid];

        // Each thread processes one vertex
        for (int edgeIdx = edgeOffsets[tid]; edgeIdx < edgeOffsets[tid + 1]; ++edgeIdx) {
            int neighbor = edgeArray[edgeIdx];
            int weight = edgeWeights[edgeIdx];

            int newDist = currentDist + weight;
            int oldDist = atomicMin(&distances[neighbor], newDist);

            if (newDist < oldDist) {
                // If we updated the distance, set the over flag
                atomicExch(over, 1);
            }
        }
    }
}

} // GraphProcessing