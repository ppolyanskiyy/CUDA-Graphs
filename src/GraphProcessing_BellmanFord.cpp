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

// Public Bellman-Ford algorithm method
std::pair<bool, std::vector<int>> Graph::bellmanFord(int startVertex, ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::Sequential:
            return bellmanFordSequential(startVertex);
        case ExecutionMode::ParallelCPU:
            return bellmanFordParallelCPU(startVertex);
        case ExecutionMode::ParallelGPU:
            return bellmanFordParallelGPU(startVertex);
        default:
            throw std::invalid_argument("Invalid ExecutionMode.");
    }
}

// Private method for sequential Bellman-Ford algorithm
std::pair<bool, std::vector<int>> Graph::bellmanFordSequential(int startVertex) {
    const int INF = std::numeric_limits<int>::max();
    int V = numVertices;
    std::vector<int> distances(V, INF);
    distances[startVertex] = 0;

    // Relax edges repeatedly
    for (int i = 1; i <= V - 1; ++i) {
        for (int u = 0; u < V; ++u) {
            for (const auto& neighbor : adjacencyList[u]) {
                int v = neighbor.first;
                int weight = neighbor.second;
                if (distances[u] != INF && distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight;
                }
            }
        }
    }

    // Check for negative-weight cycles
    for (int u = 0; u < V; ++u) {
        for (const auto& neighbor : adjacencyList[u]) {
            int v = neighbor.first;
            int weight = neighbor.second;
            if (distances[u] != INF && distances[u] + weight < distances[v]) {
                // Negative cycle detected
                return {false, distances};
            }
        }
    }

    return {true, distances};
}

// Private method for parallel Bellman-Ford algorithm using std::thread
std::pair<bool, std::vector<int>> Graph::bellmanFordParallelCPU(int startVertex) {
    const int INF = std::numeric_limits<int>::max();
    int V = numVertices;
    std::vector<std::atomic<int>> distances(V);
    for (int i = 0; i < V; ++i) {
        distances[i] = INF;
    }
    distances[startVertex] = 0;

    // Number of threads
    size_t numThreads = std::thread::hardware_concurrency();

    // Relax edges repeatedly
    for (int i = 1; i <= V - 1; ++i) {
        std::vector<std::thread> threads;
        size_t chunkSize = (V + numThreads - 1) / numThreads;

        for (size_t t = 0; t < numThreads; ++t) {
            size_t start = t * chunkSize;
            size_t end = std::min(start + chunkSize, static_cast<size_t>(V));

            threads.emplace_back([&, start, end]() {
                for (size_t u = start; u < end; ++u) {
                    int dist_u = distances[u].load();
                    if (dist_u != INF) {
                        for (const auto& neighbor : adjacencyList[u]) {
                            int v = neighbor.first;
                            int weight = neighbor.second;
                            int dist_v = distances[v].load();
                            int newDist = dist_u + weight;

                            while (newDist < dist_v) {
                                if (distances[v].compare_exchange_weak(dist_v, newDist)) {
                                    break;
                                }
                            }
                        }
                    }
                }
            });
        }

        // Join threads
        for (auto& thread : threads) {
            thread.join();
        }
    }

    // Check for negative-weight cycles
    bool hasNegativeCycle = false;
    std::vector<std::thread> threads;
    size_t chunkSize = (V + numThreads - 1) / numThreads;

    for (size_t t = 0; t < numThreads; ++t) {
        size_t start = t * chunkSize;
        size_t end = std::min(start + chunkSize, static_cast<size_t>(V));

        threads.emplace_back([&, start, end]() {
            for (size_t u = start; u < end; ++u) {
                int dist_u = distances[u].load();
                if (dist_u != INF) {
                    for (const auto& neighbor : adjacencyList[u]) {
                        int v = neighbor.first;
                        int weight = neighbor.second;
                        int dist_v = distances[v].load();
                        if (dist_u + weight < dist_v) {
                            hasNegativeCycle = true;
                            return;
                        }
                    }
                }
            }
        });
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Convert atomic distances to integer vector
    std::vector<int> result(V);
    for (int i = 0; i < V; ++i) {
        result[i] = distances[i].load();
    }

    return {!hasNegativeCycle, result};
}

// Private method for Bellman-Ford algorithm using CUDA
std::pair<bool, std::vector<int>> Graph::bellmanFordParallelGPU(int startVertex) {
    const int INF = std::numeric_limits<int>::max();
    int V = numVertices;

    // Prepare edge list
    std::vector<int> edgeSrc;
    std::vector<int> edgeDst;
    std::vector<int> edgeWeights;

    for (int u = 0; u < V; ++u) {
        for (const auto& neighbor : adjacencyList[u]) {
            int v = neighbor.first;
            int weight = neighbor.second;
            edgeSrc.push_back(u);
            edgeDst.push_back(v);
            edgeWeights.push_back(weight);
        }
    }

    int E = edgeSrc.size();

    // Device variables
    int *d_edgeSrc = nullptr;
    int *d_edgeDst = nullptr;
    int *d_edgeWeights = nullptr;
    int *d_distances = nullptr;
    int *d_updated = nullptr;
    int *d_hasNegativeCycle = nullptr;

    // Allocate device memory
    cudaMalloc(&d_edgeSrc, E * sizeof(int));
    cudaMalloc(&d_edgeDst, E * sizeof(int));
    cudaMalloc(&d_edgeWeights, E * sizeof(int));
    cudaMalloc(&d_distances, V * sizeof(int));
    cudaMalloc(&d_updated, sizeof(int));
    cudaMalloc(&d_hasNegativeCycle, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_edgeSrc, edgeSrc.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeDst, edgeDst.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeWeights, edgeWeights.data(), E * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize distances on device
    std::vector<int> h_distances(V, INF);
    h_distances[startVertex] = 0;
    cudaMemcpy(d_distances, h_distances.data(), V * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (E + threadsPerBlock - 1) / threadsPerBlock;

    // Relax edges repeatedly
    for (int i = 1; i <= V - 1; ++i) {
        int h_updated = 0;
        cudaMemcpy(d_updated, &h_updated, sizeof(int), cudaMemcpyHostToDevice);

        bellmanFordKernel<<<blocksPerGrid, threadsPerBlock>>>(
            V, E, d_edgeSrc, d_edgeDst, d_edgeWeights, d_distances, d_updated
        );

        cudaDeviceSynchronize();

        // Copy 'updated' flag from device to host
        cudaMemcpy(&h_updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost);

        // Early exit if no updates
        if (h_updated == 0) {
            break;
        }
    }

    // Check for negative-weight cycles
    int h_hasNegativeCycle = 0;
    cudaMemcpy(d_hasNegativeCycle, &h_hasNegativeCycle, sizeof(int), cudaMemcpyHostToDevice);

    bellmanFordCheckKernel<<<blocksPerGrid, threadsPerBlock>>>(
        V, E, d_edgeSrc, d_edgeDst, d_edgeWeights, d_distances, d_hasNegativeCycle
    );

    cudaDeviceSynchronize();

    // Copy 'hasNegativeCycle' flag from device to host
    cudaMemcpy(&h_hasNegativeCycle, d_hasNegativeCycle, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy distances from device to host
    cudaMemcpy(h_distances.data(), d_distances, V * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_edgeSrc);
    cudaFree(d_edgeDst);
    cudaFree(d_edgeWeights);
    cudaFree(d_distances);
    cudaFree(d_updated);
    cudaFree(d_hasNegativeCycle);

    return {h_hasNegativeCycle == 0, h_distances};
}

// Device Bellman-Ford kernel for edge relaxation
__global__
void bellmanFordKernel(
    int V,
    int E,
    const int* edgeSrc,
    const int* edgeDst,
    const int* edgeWeights,
    int* distances,
    int* updated
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < E) {
        int u = edgeSrc[tid];
        int v = edgeDst[tid];
        int weight = edgeWeights[tid];

        int dist_u = distances[u];
        int dist_v = distances[v];
        if (dist_u != INT_MAX && dist_u + weight < dist_v) {
            atomicMin(&distances[v], dist_u + weight);
            atomicExch(updated, 1);
        }
    }
}

// Device Bellman-Ford kernel to check for negative-weight cycles
__global__
void bellmanFordCheckKernel(
    int V,
    int E,
    const int* edgeSrc,
    const int* edgeDst,
    const int* edgeWeights,
    int* distances,
    int* hasNegativeCycle
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < E) {
        int u = edgeSrc[tid];
        int v = edgeDst[tid];
        int weight = edgeWeights[tid];

        int dist_u = distances[u];
        int dist_v = distances[v];
        if (dist_u != INT_MAX && dist_u + weight < dist_v) {
            atomicExch(hasNegativeCycle, 1);
        }
    }
}

} // namespace GraphProcessing
