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

// Public Floyd-Warshall algorithm method
std::vector<std::vector<int>> Graph::floydWarshall(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::Sequential:
            return floydWarshallSequential();
        case ExecutionMode::ParallelCPU:
            return floydWarshallParallelCPU();
        case ExecutionMode::ParallelGPU:
            return floydWarshallParallelGPU();
        default:
            throw std::invalid_argument("Invalid ExecutionMode.");
    }
}

// Private method for sequential Floyd-Warshall algorithm
std::vector<std::vector<int>> Graph::floydWarshallSequential() {
    const int INF = std::numeric_limits<int>::max();
    int V = numVertices;
    std::vector<std::vector<int>> dist(V, std::vector<int>(V, INF));

    // Initialize distances
    for (int i = 0; i < V; ++i) {
        dist[i][i] = 0;
        for (const auto& neighbor : adjacencyList[i]) {
            int j = neighbor.first;
            int weight = neighbor.second;
            dist[i][j] = weight;
        }
    }

    // Floyd-Warshall algorithm
    for (int k = 0; k < V; ++k) {
        for (int i = 0; i < V; ++i) {
            for (int j = 0; j < V; ++j) {
                if (dist[i][k] != INF && dist[k][j] != INF &&
                    dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    return dist;
}

// Private method for parallel Floyd-Warshall algorithm using std::thread
std::vector<std::vector<int>> Graph::floydWarshallParallelCPU() {
    const int INF = std::numeric_limits<int>::max();
    int V = numVertices;
    std::vector<std::vector<std::atomic<int>>> dist(V, std::vector<std::atomic<int>>(V));

    // Initialize distances
    for (int i = 0; i < V; ++i) {
        dist[i][i] = 0;
        for (int j = 0; j < V; ++j) {
            if (i != j) {
                dist[i][j] = INF;
            }
        }
        for (const auto& neighbor : adjacencyList[i]) {
            int j = neighbor.first;
            int weight = neighbor.second;
            dist[i][j] = weight;
        }
    }

    // Number of threads
    size_t numThreads = std::thread::hardware_concurrency();

    // Floyd-Warshall algorithm with parallelization over 'i'
    for (int k = 0; k < V; ++k) {
        std::vector<std::thread> threads;

        // Divide 'i' among threads
        size_t chunkSize = (V + numThreads - 1) / numThreads;

        for (size_t t = 0; t < numThreads; ++t) {
            size_t start = t * chunkSize;
            size_t end = std::min(start + chunkSize, static_cast<size_t>(V));

            threads.emplace_back([&, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    for (int j = 0; j < V; ++j) {
                        int dist_ik = dist[i][k].load();
                        int dist_kj = dist[k][j].load();
                        int dist_ij = dist[i][j].load();

                        if (dist_ik != INF && dist_kj != INF &&
                            dist_ik + dist_kj < dist_ij) {
                            dist[i][j] = dist_ik + dist_kj;
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

    // Convert atomic distances to integer matrix
    std::vector<std::vector<int>> result(V, std::vector<int>(V));
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            result[i][j] = dist[i][j].load();
        }
    }

    return result;
}

// Private method for Floyd-Warshall algorithm using CUDA
std::vector<std::vector<int>> Graph::floydWarshallParallelGPU() {
    const int INF = std::numeric_limits<int>::max();
    int V = numVertices;

    // Flatten the distance matrix
    std::vector<int> dist(V * V, INF);

    // Initialize distances
    for (int i = 0; i < V; ++i) {
        dist[i * V + i] = 0;
        for (const auto& neighbor : adjacencyList[i]) {
            int j = neighbor.first;
            int weight = neighbor.second;
            dist[i * V + j] = weight;
        }
    }

    // Device variable
    int* d_dist = nullptr;

    // Allocate device memory
    cudaMalloc(&d_dist, V * V * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_dist, dist.data(), V * V * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel execution parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((V + blockSize.x - 1) / blockSize.x, (V + blockSize.y - 1) / blockSize.y);

    // Run the kernel
    for (int k = 0; k < V; ++k) {
        floydWarshallKernel<<<gridSize, blockSize>>>(d_dist, V, k);
        cudaDeviceSynchronize();
    }

    // Copy result back to host
    cudaMemcpy(dist.data(), d_dist, V * V * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_dist);

    // Convert flattened distance matrix to 2D vector
    std::vector<std::vector<int>> result(V, std::vector<int>(V));
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            result[i][j] = dist[i * V + j];
        }
    }

    return result;
}

// Device Floyd-Warshall kernel with shared memory and memory coalescing
__global__
void floydWarshallKernel(int* dist, int V, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < V && j < V) {
        int ij = dist[i * V + j];
        int ik = dist[i * V + k];
        int kj = dist[k * V + j];

        if (ik != INT_MAX && kj != INT_MAX && ik + kj < ij) {
            dist[i * V + j] = ik + kj;
        }
    }
}

} // namespace GraphProcessing
