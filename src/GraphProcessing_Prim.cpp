// GraphProcessing.cpp

#include "GraphProcessing.h"
#include <vector>
#include <queue>
#include <limits>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace GraphProcessing {

// Public Prim's algorithm method
std::vector<std::pair<int, int>> Graph::prim(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::Sequential:
            return primSequential();
        case ExecutionMode::ParallelCPU:
            return primParallelCPU();
        case ExecutionMode::ParallelGPU:
            return primParallelGPU();
        default:
            throw std::invalid_argument("Invalid ExecutionMode.");
    }
}

// Private method for sequential Prim's algorithm
std::vector<std::pair<int, int>> Graph::primSequential() {
    const int INF = std::numeric_limits<int>::max();
    int V = numVertices;

    std::vector<int> key(V, INF);             // Minimum weight edge to reach vertex
    std::vector<int> parent(V, -1);           // Parent of each vertex in the MST
    std::vector<bool> inMST(V, false);        // Whether vertex is included in MST

    // Min-heap priority queue: (key, vertex)
    using pii = std::pair<int, int>;
    std::priority_queue<pii, std::vector<pii>, std::greater<pii>> minHeap;

    // Start from vertex 0
    key[0] = 0;
    minHeap.emplace(0, 0);

    while (!minHeap.empty()) {
        int u = minHeap.top().second;
        minHeap.pop();

        if (inMST[u]) continue;

        inMST[u] = true;

        for (const auto& neighbor : adjacencyList[u]) {
            int v = neighbor.first;
            int weight = neighbor.second;

            if (!inMST[v] && weight < key[v]) {
                key[v] = weight;
                parent[v] = u;
                minHeap.emplace(key[v], v);
            }
        }
    }

    // Collect edges in the MST
    std::vector<std::pair<int, int>> mstEdges;
    for (int v = 1; v < V; ++v) {
        if (parent[v] != -1) {
            mstEdges.emplace_back(parent[v], v);
        }
    }

    return mstEdges;
}

// Private method for parallel CPU Prim's algorithm
std::vector<std::pair<int, int>> Graph::primParallelCPU() {
    const int INF = std::numeric_limits<int>::max();
    int V = numVertices;

    std::vector<std::atomic<int>> key(V);
    std::vector<std::atomic<int>> parent(V);
    std::vector<std::atomic<bool>> inMST(V);

    for (int i = 0; i < V; ++i) {
        key[i] = INF;
        parent[i] = -1;
        inMST[i] = false;
    }

    // Shared priority queue protected by mutex
    using pii = std::pair<int, int>;
    auto cmp = [](const pii& left, const pii& right) {
        return left.first > right.first;
    };
    std::priority_queue<pii, std::vector<pii>, decltype(cmp)> minHeap(cmp);
    std::mutex minHeapMutex;

    // Start from vertex 0
    key[0] = 0;
    {
        std::lock_guard<std::mutex> lock(minHeapMutex);
        minHeap.emplace(0, 0);
    }

    std::vector<std::thread> threads;
    size_t numThreads = std::thread::hardware_concurrency();
    std::atomic<int> verticesProcessed(0);

    auto worker = [&]() {
        while (verticesProcessed.load() < V) {
            int u = -1;
            {
                std::lock_guard<std::mutex> lock(minHeapMutex);
                if (!minHeap.empty()) {
                    u = minHeap.top().second;
                    minHeap.pop();
                }
            }

            if (u == -1) continue;

            // Atomically check and set inMST[u]
            bool expected = false;
            if (!inMST[u].compare_exchange_strong(expected, true)) {
                continue;
            }

            verticesProcessed.fetch_add(1);

            for (const auto& neighbor : adjacencyList[u]) {
                int v = neighbor.first;
                int weight = neighbor.second;

                if (!inMST[v]) {
                    int currentKey = key[v].load();
                    if (weight < currentKey) {
                        if (key[v].compare_exchange_strong(currentKey, weight)) {
                            parent[v] = u;
                            std::lock_guard<std::mutex> lock(minHeapMutex);
                            minHeap.emplace(key[v], v);
                        }
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

    // Collect edges in the MST
    std::vector<std::pair<int, int>> mstEdges;
    for (int v = 1; v < V; ++v) {
        if (parent[v] != -1) {
            mstEdges.emplace_back(parent[v], v);
        }
    }

    return mstEdges;
}

// Private method for Prim's algorithm using CUDA
std::vector<std::pair<int, int>> Graph::primParallelGPU() {
    const int INF = std::numeric_limits<int>::max();
    int V = numVertices;

    // Prepare adjacency list in CSR format
    std::vector<int> edgeArray;      // Flattened edge list
    std::vector<int> edgeWeights;    // Corresponding edge weights
    std::vector<int> edgeOffsets;    // Index offsets for each vertex
    edgeOffsets.resize(V + 1);

    int totalEdges = 0;
    for (int i = 0; i < V; ++i) {
        edgeOffsets[i] = totalEdges;
        const auto& neighbors = adjacencyList[i];
        totalEdges += neighbors.size();
        for (const auto& neighbor : neighbors) {
            edgeArray.push_back(neighbor.first);
            edgeWeights.push_back(neighbor.second);
        }
    }
    edgeOffsets[V] = totalEdges;

    // Device variables
    int *d_edgeArray = nullptr;
    int *d_edgeWeights = nullptr;
    int *d_edgeOffsets = nullptr;
    int *d_key = nullptr;
    int *d_parent = nullptr;
    bool *d_inMST = nullptr;
    bool *d_done = nullptr;

    // Allocate device memory
    cudaMalloc(&d_edgeArray, edgeArray.size() * sizeof(int));
    cudaMalloc(&d_edgeWeights, edgeWeights.size() * sizeof(int));
    cudaMalloc(&d_edgeOffsets, edgeOffsets.size() * sizeof(int));
    cudaMalloc(&d_key, V * sizeof(int));
    cudaMalloc(&d_parent, V * sizeof(int));
    cudaMalloc(&d_inMST, V * sizeof(bool));
    cudaMalloc(&d_done, sizeof(bool));

    // Copy data to device
    cudaMemcpy(d_edgeArray, edgeArray.data(), edgeArray.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeWeights, edgeWeights.data(), edgeWeights.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeOffsets, edgeOffsets.data(), edgeOffsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize device arrays
    std::vector<int> h_key(V, INF);
    std::vector<int> h_parent(V, -1);
    std::vector<bool> h_inMST(V, false);
    h_key[0] = 0; // Start from vertex 0

    cudaMemcpy(d_key, h_key.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent, h_parent.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inMST, h_inMST.data(), V * sizeof(bool), cudaMemcpyHostToDevice);

    // Kernel execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (V + threadsPerBlock - 1) / threadsPerBlock;

    bool h_done = false;

    while (!h_done) {
        h_done = true;
        cudaMemcpy(d_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice);

        // Launch Prim's kernel
        primKernel<<<blocksPerGrid, threadsPerBlock>>>(
            V,
            d_edgeArray,
            d_edgeWeights,
            d_edgeOffsets,
            d_key,
            d_parent,
            d_inMST,
            d_done
        );

        cudaDeviceSynchronize();

        // Copy 'done' flag from device to host
        cudaMemcpy(&h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    // Copy results from device to host
    cudaMemcpy(h_parent.data(), d_parent, V * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_edgeArray);
    cudaFree(d_edgeWeights);
    cudaFree(d_edgeOffsets);
    cudaFree(d_key);
    cudaFree(d_parent);
    cudaFree(d_inMST);
    cudaFree(d_done);

    // Collect edges in the MST
    std::vector<std::pair<int, int>> mstEdges;
    for (int v = 1; v < V; ++v) {
        if (h_parent[v] != -1) {
            mstEdges.emplace_back(h_parent[v], v);
        }
    }

    return mstEdges;
}

// Device Prim's kernel
__global__
void primKernel(
    int V,
    const int* edgeArray,
    const int* edgeWeights,
    const int* edgeOffsets,
    int* key,
    int* parent,
    bool* inMST,
    bool* done
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < V && !inMST[tid]) {
        // Find the minimum key value among all vertices not yet included in MST
        __shared__ int minKey;
        __shared__ int minIndex;

        if (threadIdx.x == 0) {
            minKey = INT_MAX;
            minIndex = -1;
        }
        __syncthreads();

        int myKey = key[tid];

        // Atomic min reduction to find the vertex with the minimum key value
        atomicMin(&minKey, myKey);

        __syncthreads();

        if (myKey == minKey && !inMST[tid]) {
            inMST[tid] = true;
            minIndex = tid;
        }

        __syncthreads();

        if (minIndex != -1) {
            int u = minIndex;

            // Update key and parent of all adjacent vertices
            int edgeStart = edgeOffsets[u];
            int edgeEnd = edgeOffsets[u + 1];

            for (int edgeIdx = edgeStart; edgeIdx < edgeEnd; ++edgeIdx) {
                int v = edgeArray[edgeIdx];
                int weight = edgeWeights[edgeIdx];

                if (!inMST[v] && weight < key[v]) {
                    key[v] = weight;
                    parent[v] = u;
                    *done = false;
                }
            }
        }
    }
}

} // namespace GraphProcessing
