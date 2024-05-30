#include "GraphTraversal.h"


void bfsSequential(const std::vector<std::vector<int>> & graph, int start) {
    int V = graph.size();
    std::vector<bool> visited(V, false);
    std::queue<int> queue;

    visited[start] = true;
    queue.push(start);

    while (!queue.empty()) {
        int vertex = queue.front();
        //std::cout << vertex << " ";
        queue.pop();

        for (int neighbor : graph[vertex]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }
}


// CUDA Kernel for BFS
__global__ void bfsKernelDefault(int* d_graph, int* d_queue, bool* d_visited, int* d_frontierSize, int V) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < V && d_queue[idx] != INF) {
        int vertex = d_queue[idx];
        d_queue[idx] = INF;

        for (int i = 0; i < V; i++) {
            if (d_graph[vertex * V + i] == 1 && !d_visited[i]) {
                d_visited[i] = true;
                int pos = atomicAdd(d_frontierSize, 1);
                d_queue[V + pos] = i;
            }
        }
    }
}

// Host function to launch BFS on GPU
void bfsDefaultCUDA(const std::vector<std::vector<int>> & graph, int start) {
    int V = graph.size();

    // Flatten the graph adjacency matrix
    std::vector<int> flatGraph(V * V, 0);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            flatGraph[i * V + j] = graph[i][j];
        }
    }

    // Device memory allocation
    int* d_graph, * d_queue, * d_frontierSize;
    bool* d_visited;
    cudaMalloc((void**)&d_graph, V * V * sizeof(int));
    cudaMalloc((void**)&d_queue, 2 * V * sizeof(int));
    cudaMalloc((void**)&d_visited, V * sizeof(bool));
    cudaMalloc((void**)&d_frontierSize, sizeof(int));

    // Host memory allocation
    std::vector<int> h_queue(2 * V, INF);
    std::vector<char> h_visited(V, 0); // Використання std::vector<char> для відвіданих вершин
    int h_frontierSize = 0;

    // Initialize the queue with the start vertex
    h_queue[0] = start;
    h_visited[start] = true;
    h_frontierSize = 1;

    // Copy data to device
    cudaMemcpy(d_graph, flatGraph.data(), V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queue, h_queue.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited.data(), V * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontierSize, &h_frontierSize, sizeof(int), cudaMemcpyHostToDevice);

    // Launch BFS kernel
    int numBlocks = (V + 255) / 256;
    int numThreads = 256;

    while (h_frontierSize > 0) {
        cudaMemcpy(d_frontierSize, &h_frontierSize, sizeof(int), cudaMemcpyHostToDevice);
        bfsKernelDefault << <numBlocks, numThreads >> > (d_graph, d_queue, d_visited, d_frontierSize, V);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_frontierSize, d_frontierSize, sizeof(int), cudaMemcpyDeviceToHost);

        // Shift the queue
        cudaMemcpy(h_queue.data(), d_queue, 2 * V * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < V; ++i) {
            h_queue[i] = h_queue[V + i];
            h_queue[V + i] = INF;
        }
        cudaMemcpy(d_queue, h_queue.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Copy visited array back to host
    cudaMemcpy(h_visited.data(), d_visited, V * sizeof(char), cudaMemcpyDeviceToHost);

    // Print the BFS result
    //std::cout << "BFS Order: ";
    //for (int i = 0; i < V; ++i) {
    //    if (h_visited[i]) {
    //        std::cout << i << " ";
    //    }
    //}
    //std::cout << std::endl;

    // Free device memory
    cudaFree(d_graph);
    cudaFree(d_queue);
    cudaFree(d_visited);
    cudaFree(d_frontierSize);
}


#define SHARED_MEM_SIZE 256


// CUDA Kernel for BFS with shared memory
__global__ void bfsKernelShared(int* d_graph, int* d_queue, bool* d_visited, int* d_frontierSize, int V) {
    __shared__ int sharedQueue[SHARED_MEM_SIZE];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < V && d_queue[idx] != INF) {
        int vertex = d_queue[idx];
        d_queue[idx] = INF;

        int threadQueuePos = 0;
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            if (d_graph[vertex * V + i] == 1 && !d_visited[i]) {
                d_visited[i] = true;
                int pos = atomicAdd(&threadQueuePos, 1);
                if (pos < SHARED_MEM_SIZE) {
                    sharedQueue[pos] = i;
                }
                else {
                    int globalPos = atomicAdd(d_frontierSize, 1);
                    d_queue[V + globalPos] = i;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < threadQueuePos && threadIdx.x < SHARED_MEM_SIZE) {
            int globalPos = atomicAdd(d_frontierSize, 1);
            d_queue[V + globalPos] = sharedQueue[threadIdx.x];
        }
    }
}

// Host function to launch BFS on GPU
void bfsSharedCUDA(const std::vector<std::vector<int>> & graph, int start) {
    int V = graph.size();

    // Flatten the graph adjacency matrix
    std::vector<int> flatGraph(V * V, 0);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            flatGraph[i * V + j] = graph[i][j];
        }
    }

    // Device memory allocation
    int* d_graph, * d_queue, * d_frontierSize;
    bool* d_visited;
    cudaMalloc((void**)&d_graph, V * V * sizeof(int));
    cudaMalloc((void**)&d_queue, 2 * V * sizeof(int));
    cudaMalloc((void**)&d_visited, V * sizeof(bool));
    cudaMalloc((void**)&d_frontierSize, sizeof(int));

    // Host memory allocation
    std::vector<int> h_queue(2 * V, INF);
    std::vector<char> h_visited(V, 0); // Використання std::vector<char> для відвіданих вершин
    int h_frontierSize = 0;

    // Initialize the queue with the start vertex
    h_queue[0] = start;
    h_visited[start] = true;
    h_frontierSize = 1;

    // Copy data to device
    cudaMemcpy(d_graph, flatGraph.data(), V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queue, h_queue.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited.data(), V * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontierSize, &h_frontierSize, sizeof(int), cudaMemcpyHostToDevice);

    // Launch BFS kernel
    int numBlocks = (V + 255) / 256;
    int numThreads = 256;

    while (h_frontierSize > 0) {
        cudaMemcpy(d_frontierSize, &h_frontierSize, sizeof(int), cudaMemcpyHostToDevice);
        bfsKernelShared << <numBlocks, numThreads >> > (d_graph, d_queue, d_visited, d_frontierSize, V);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_frontierSize, d_frontierSize, sizeof(int), cudaMemcpyDeviceToHost);

        // Shift the queue
        cudaMemcpy(h_queue.data(), d_queue, 2 * V * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < V; ++i) {
            h_queue[i] = h_queue[V + i];
            h_queue[V + i] = INF;
        }
        cudaMemcpy(d_queue, h_queue.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Copy visited array back to host
    cudaMemcpy(h_visited.data(), d_visited, V * sizeof(char), cudaMemcpyDeviceToHost);

    //// Print the BFS result
    //std::cout << "BFS Order: ";
    //for (int i = 0; i < V; ++i) {
    //    if (h_visited[i]) {
    //        std::cout << i << " ";
    //    }
    //}
    //std::cout << std::endl;

    // Free device memory
    cudaFree(d_graph);
    cudaFree(d_queue);
    cudaFree(d_visited);
    cudaFree(d_frontierSize);
}


// CUDA Kernel for BFS with memory coalescing
__global__ void bfsKernelCoalesced(int* d_graph, int* d_queue, bool* d_visited, int* d_frontierSize, int V) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < V && d_queue[idx] != INF) {
        int vertex = d_queue[idx];
        d_queue[idx] = INF;

        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            if (d_graph[vertex * V + i] == 1 && !d_visited[i]) {
                d_visited[i] = true;
                int pos = atomicAdd(d_frontierSize, 1);
                d_queue[V + pos] = i;
            }
        }
    }
}

// Host function to launch BFS on GPU
void bfsCoalescedCUDA(const std::vector<std::vector<int>> &graph, int start) {
    int V = graph.size();

    // Flatten the graph adjacency matrix
    std::vector<int> flatGraph(V * V, 0);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            flatGraph[i * V + j] = graph[i][j];
        }
    }

    // Device memory allocation
    int* d_graph, * d_queue, * d_frontierSize;
    bool* d_visited;
    cudaMalloc((void**)&d_graph, V * V * sizeof(int));
    cudaMalloc((void**)&d_queue, 2 * V * sizeof(int));
    cudaMalloc((void**)&d_visited, V * sizeof(bool));
    cudaMalloc((void**)&d_frontierSize, sizeof(int));

    // Host memory allocation
    std::vector<int> h_queue(2 * V, INF);
    std::vector<char> h_visited(V, 0); // Використання std::vector<char> для відвіданих вершин
    int h_frontierSize = 0;

    // Initialize the queue with the start vertex
    h_queue[0] = start;
    h_visited[start] = true;
    h_frontierSize = 1;

    // Copy data to device
    cudaMemcpy(d_graph, flatGraph.data(), V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queue, h_queue.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited.data(), V * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontierSize, &h_frontierSize, sizeof(int), cudaMemcpyHostToDevice);

    // Launch BFS kernel
    int numBlocks = (V + 255) / 256;
    int numThreads = 256;

    while (h_frontierSize > 0) {
        cudaMemcpy(d_frontierSize, &h_frontierSize, sizeof(int), cudaMemcpyHostToDevice);
        bfsKernelCoalesced << <numBlocks, numThreads >> > (d_graph, d_queue, d_visited, d_frontierSize, V);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_frontierSize, d_frontierSize, sizeof(int), cudaMemcpyDeviceToHost);

        // Shift the queue
        cudaMemcpy(h_queue.data(), d_queue, 2 * V * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < V; ++i) {
            h_queue[i] = h_queue[V + i];
            h_queue[V + i] = INF;
        }
        cudaMemcpy(d_queue, h_queue.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Copy visited array back to host
    cudaMemcpy(h_visited.data(), d_visited, V * sizeof(char), cudaMemcpyDeviceToHost);

    // Print the BFS result
    //std::cout << "BFS Order: ";
    //for (int i = 0; i < V; ++i) {
    //    if (h_visited[i]) {
    //        std::cout << i << " ";
    //    }
    //}
    //std::cout << std::endl;

    // Free device memory
    cudaFree(d_graph);
    cudaFree(d_queue);
    cudaFree(d_visited);
    cudaFree(d_frontierSize);
}


// CUDA Kernel for BFS with shared memory and memory coalescing
__global__ void bfsKernelFullyOptimized(int* d_graph, int* d_queue, bool* d_visited, int* d_frontierSize, int V) {
    __shared__ int sharedQueue[SHARED_MEM_SIZE];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < V && d_queue[idx] != INF) {
        int vertex = d_queue[idx];
        d_queue[idx] = INF;

        int threadQueuePos = 0;
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            if (d_graph[vertex * V + i] == 1 && !d_visited[i]) {
                d_visited[i] = true;
                int pos = atomicAdd(&threadQueuePos, 1);
                if (pos < SHARED_MEM_SIZE) {
                    sharedQueue[pos] = i;
                }
                else {
                    int globalPos = atomicAdd(d_frontierSize, 1);
                    d_queue[V + globalPos] = i;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < threadQueuePos && threadIdx.x < SHARED_MEM_SIZE) {
            int globalPos = atomicAdd(d_frontierSize, 1);
            d_queue[V + globalPos] = sharedQueue[threadIdx.x];
        }
    }
}

// Host function to launch BFS on GPU
void bfsFullyOptimizedCUDA(const std::vector<std::vector<int>> & graph, int start) {
    int V = graph.size();

    // Flatten the graph adjacency matrix
    std::vector<int> flatGraph(V * V, 0);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            flatGraph[i * V + j] = graph[i][j];
        }
    }

    // Device memory allocation
    int* d_graph, * d_queue, * d_frontierSize;
    bool* d_visited;
    cudaMalloc((void**)&d_graph, V * V * sizeof(int));
    cudaMalloc((void**)&d_queue, 2 * V * sizeof(int));
    cudaMalloc((void**)&d_visited, V * sizeof(bool));
    cudaMalloc((void**)&d_frontierSize, sizeof(int));

    // Host memory allocation
    std::vector<int> h_queue(2 * V, INF);
    std::vector<char> h_visited(V, 0); // Використання std::vector<char> для відвіданих вершин
    int h_frontierSize = 0;

    // Initialize the queue with the start vertex
    h_queue[0] = start;
    h_visited[start] = true;
    h_frontierSize = 1;

    // Copy data to device
    cudaMemcpy(d_graph, flatGraph.data(), V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queue, h_queue.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited.data(), V * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontierSize, &h_frontierSize, sizeof(int), cudaMemcpyHostToDevice);

    // Launch BFS kernel
    int numBlocks = (V + 255) / 256;
    int numThreads = 256;

    while (h_frontierSize > 0) {
        cudaMemcpy(d_frontierSize, &h_frontierSize, sizeof(int), cudaMemcpyHostToDevice);
        bfsKernelFullyOptimized << <numBlocks, numThreads >> > (d_graph, d_queue, d_visited, d_frontierSize, V);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_frontierSize, d_frontierSize, sizeof(int), cudaMemcpyDeviceToHost);

        // Shift the queue
        cudaMemcpy(h_queue.data(), d_queue, 2 * V * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < V; ++i) {
            h_queue[i] = h_queue[V + i];
            h_queue[V + i] = INF;
        }
        cudaMemcpy(d_queue, h_queue.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Copy visited array back to host
    cudaMemcpy(h_visited.data(), d_visited, V * sizeof(char), cudaMemcpyDeviceToHost);

    // Print the BFS result
    //std::cout << "BFS Order: ";
    //for (int i = 0; i < V; ++i) {
    //    if (h_visited[i]) {
    //        std::cout << i << " ";
    //    }
    //}
    //std::cout << std::endl;

    // Free device memory
    cudaFree(d_graph);
    cudaFree(d_queue);
    cudaFree(d_visited);
    cudaFree(d_frontierSize);
}