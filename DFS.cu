#include "GraphTraversal.h"


void dfsSequential(const std::vector<std::vector<int>>& adj, int start) {
    int V = adj.size();
    std::vector<bool> visited(V, false);
    std::stack<int> stack;

    stack.push(start);

    while (!stack.empty()) {
        int vertex = stack.top();
        stack.pop();

        if (!visited[vertex]) {
            //std::cout << vertex << " ";
            visited[vertex] = true;
        }

        for (int neighbor : adj[vertex]) {
            if (!visited[neighbor]) {
                stack.push(neighbor);
            }
        }
    }
}

// CUDA Kernel for DFS
__global__ void dfsDefaultKernel(int* d_graph, int* d_stack, bool* d_visited, int* d_stackSize, int V) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < V && d_stack[idx] != INF) {
        int vertex = d_stack[idx];
        d_stack[idx] = INF;

        for (int i = 0; i < V; ++i) {
            if (d_graph[vertex * V + i] == 1 && !d_visited[i]) {
                d_visited[i] = true;
                int pos = atomicAdd(d_stackSize, 1);
                d_stack[V + pos] = i;
            }
        }
    }
}

// Host function to launch DFS on GPU
void dfsDefaultCUDA(const std::vector<std::vector<int>>& graph, int start) {
    int V = graph.size();

    // Flatten the graph adjacency matrix
    std::vector<int> flatGraph(V * V, 0);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            flatGraph[i * V + j] = graph[i][j];
        }
    }

    // Device memory allocation
    int* d_graph, * d_stack, * d_stackSize;
    bool* d_visited;
    cudaMalloc((void**)&d_graph, V * V * sizeof(int));
    cudaMalloc((void**)&d_stack, 2 * V * sizeof(int));
    cudaMalloc((void**)&d_visited, V * sizeof(bool));
    cudaMalloc((void**)&d_stackSize, sizeof(int));

    // Host memory allocation
    std::vector<int> h_stack(2 * V, INF);
    std::vector<char> h_visited(V, 0); // Використання std::vector<char> для відвіданих вершин
    int h_stackSize = 0;

    // Initialize the stack with the start vertex
    h_stack[0] = start;
    h_visited[start] = true;
    h_stackSize = 1;

    // Copy data to device
    cudaMemcpy(d_graph, flatGraph.data(), V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stack, h_stack.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited.data(), V * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stackSize, &h_stackSize, sizeof(int), cudaMemcpyHostToDevice);

    // Launch DFS kernel
    int numBlocks = (V + 255) / 256;
    int numThreads = 256;

    while (h_stackSize > 0) {
        cudaMemcpy(d_stackSize, &h_stackSize, sizeof(int), cudaMemcpyHostToDevice);
        dfsDefaultKernel << <numBlocks, numThreads >> > (d_graph, d_stack, d_visited, d_stackSize, V);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_stackSize, d_stackSize, sizeof(int), cudaMemcpyDeviceToHost);

        // Shift the stack
        cudaMemcpy(h_stack.data(), d_stack, 2 * V * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < V; ++i) {
            h_stack[i] = h_stack[V + i];
            h_stack[V + i] = INF;
        }
        cudaMemcpy(d_stack, h_stack.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Copy visited array back to host
    cudaMemcpy(h_visited.data(), d_visited, V * sizeof(char), cudaMemcpyDeviceToHost);

    // Print the DFS result
    //std::cout << "DFS Order: ";
    //for (int i = 0; i < V; ++i) {
    //    if (h_visited[i]) {
    //        std::cout << i << " ";
    //    }
    //}
    //std::cout << std::endl;

    // Free device memory
    cudaFree(d_graph);
    cudaFree(d_stack);
    cudaFree(d_visited);
    cudaFree(d_stackSize);
}

// CUDA Kernel for DFS with shared memory
__global__ void dfsKernelShared(int* d_graph, int* d_stack, bool* d_visited, int* d_stackSize, int V) {
    __shared__ int sharedStack[SHARED_MEM_SIZE];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < V && d_stack[idx] != INF) {
        int vertex = d_stack[idx];
        d_stack[idx] = INF;

        int threadStackPos = 0;
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            if (d_graph[vertex * V + i] == 1 && !d_visited[i]) {
                d_visited[i] = true;
                int pos = atomicAdd(&threadStackPos, 1);
                if (pos < SHARED_MEM_SIZE) {
                    sharedStack[pos] = i;
                }
                else {
                    int globalPos = atomicAdd(d_stackSize, 1);
                    d_stack[V + globalPos] = i;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < threadStackPos && threadIdx.x < SHARED_MEM_SIZE) {
            int globalPos = atomicAdd(d_stackSize, 1);
            d_stack[V + globalPos] = sharedStack[threadIdx.x];
        }
    }
}

// Host function to launch DFS on GPU
void dfsSharedCUDA(const std::vector<std::vector<int>>& graph, int start) {
    int V = graph.size();

    // Flatten the graph adjacency matrix
    std::vector<int> flatGraph(V * V, 0);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            flatGraph[i * V + j] = graph[i][j];
        }
    }

    // Device memory allocation
    int* d_graph, * d_stack, * d_stackSize;
    bool* d_visited;
    cudaMalloc((void**)&d_graph, V * V * sizeof(int));
    cudaMalloc((void**)&d_stack, 2 * V * sizeof(int));
    cudaMalloc((void**)&d_visited, V * sizeof(bool));
    cudaMalloc((void**)&d_stackSize, sizeof(int));

    // Host memory allocation
    std::vector<int> h_stack(2 * V, INF);
    std::vector<char> h_visited(V, 0); // Використання std::vector<char> для відвіданих вершин
    int h_stackSize = 0;

    // Initialize the stack with the start vertex
    h_stack[0] = start;
    h_visited[start] = true;
    h_stackSize = 1;

    // Copy data to device
    cudaMemcpy(d_graph, flatGraph.data(), V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stack, h_stack.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited.data(), V * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stackSize, &h_stackSize, sizeof(int), cudaMemcpyHostToDevice);

    // Launch DFS kernel
    int numBlocks = (V + 255) / 256;
    int numThreads = 256;

    while (h_stackSize > 0) {
        cudaMemcpy(d_stackSize, &h_stackSize, sizeof(int), cudaMemcpyHostToDevice);
        dfsKernelShared << <numBlocks, numThreads >> > (d_graph, d_stack, d_visited, d_stackSize, V);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_stackSize, d_stackSize, sizeof(int), cudaMemcpyDeviceToHost);

        // Shift the stack
        cudaMemcpy(h_stack.data(), d_stack, 2 * V * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < V; ++i) {
            h_stack[i] = h_stack[V + i];
            h_stack[V + i] = INF;
        }
        cudaMemcpy(d_stack, h_stack.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Copy visited array back to host
    cudaMemcpy(h_visited.data(), d_visited, V * sizeof(char), cudaMemcpyDeviceToHost);

    // Print the DFS result
    //std::cout << "DFS Order: ";
    //for (int i = 0; i < V; ++i) {
    //    if (h_visited[i]) {
    //        std::cout << i << " ";
    //    }
    //}
    //std::cout << std::endl;

    // Free device memory
    cudaFree(d_graph);
    cudaFree(d_stack);
    cudaFree(d_visited);
    cudaFree(d_stackSize);
}


// CUDA Kernel for DFS with memory coalescing
__global__ void dfsKernelCoalesced(int* d_graph, int* d_stack, bool* d_visited, int* d_stackSize, int V) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < V && d_stack[idx] != INF) {
        int vertex = d_stack[idx];
        d_stack[idx] = INF;

        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            if (d_graph[vertex * V + i] == 1 && !d_visited[i]) {
                d_visited[i] = true;
                int pos = atomicAdd(d_stackSize, 1);
                d_stack[V + pos] = i;
            }
        }
    }
}

// Host function to launch DFS on GPU
void dfsCoalescedCUDA(const std::vector<std::vector<int>>& graph, int start) {
    int V = graph.size();

    // Flatten the graph adjacency matrix
    std::vector<int> flatGraph(V * V, 0);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            flatGraph[i * V + j] = graph[i][j];
        }
    }

    // Device memory allocation
    int* d_graph, * d_stack, * d_stackSize;
    bool* d_visited;
    cudaMalloc((void**)&d_graph, V * V * sizeof(int));
    cudaMalloc((void**)&d_stack, 2 * V * sizeof(int));
    cudaMalloc((void**)&d_visited, V * sizeof(bool));
    cudaMalloc((void**)&d_stackSize, sizeof(int));

    // Host memory allocation
    std::vector<int> h_stack(2 * V, INF);
    std::vector<char> h_visited(V, 0); // Використання std::vector<char> для відвіданих вершин
    int h_stackSize = 0;

    // Initialize the stack with the start vertex
    h_stack[0] = start;
    h_visited[start] = true;
    h_stackSize = 1;

    // Copy data to device
    cudaMemcpy(d_graph, flatGraph.data(), V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stack, h_stack.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited.data(), V * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stackSize, &h_stackSize, sizeof(int), cudaMemcpyHostToDevice);

    // Launch DFS kernel
    int numBlocks = (V + 255) / 256;
    int numThreads = 256;

    while (h_stackSize > 0) {
        cudaMemcpy(d_stackSize, &h_stackSize, sizeof(int), cudaMemcpyHostToDevice);
        dfsKernelCoalesced << <numBlocks, numThreads >> > (d_graph, d_stack, d_visited, d_stackSize, V);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_stackSize, d_stackSize, sizeof(int), cudaMemcpyDeviceToHost);

        // Shift the stack
        cudaMemcpy(h_stack.data(), d_stack, 2 * V * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < V; ++i) {
            h_stack[i] = h_stack[V + i];
            h_stack[V + i] = INF;
        }
        cudaMemcpy(d_stack, h_stack.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Copy visited array back to host
    cudaMemcpy(h_visited.data(), d_visited, V * sizeof(char), cudaMemcpyDeviceToHost);

    // Print the DFS result
    //std::cout << "DFS Order: ";
    //for (int i = 0; i < V; ++i) {
    //    if (h_visited[i]) {
    //        std::cout << i << " ";
    //    }
    //}
    //std::cout << std::endl;

    // Free device memory
    cudaFree(d_graph);
    cudaFree(d_stack);
    cudaFree(d_visited);
    cudaFree(d_stackSize);
}


// CUDA Kernel for DFS with shared memory and memory coalescing
__global__ void dfsKernelFullyOptimized(int* d_graph, int* d_stack, bool* d_visited, int* d_stackSize, int V) {
    __shared__ int sharedStack[SHARED_MEM_SIZE];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < V && d_stack[idx] != INF) {
        int vertex = d_stack[idx];
        d_stack[idx] = INF;

        int threadStackPos = 0;
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            if (d_graph[vertex * V + i] == 1 && !d_visited[i]) {
                d_visited[i] = true;
                int pos = atomicAdd(&threadStackPos, 1);
                if (pos < SHARED_MEM_SIZE) {
                    sharedStack[pos] = i;
                }
                else {
                    int globalPos = atomicAdd(d_stackSize, 1);
                    d_stack[V + globalPos] = i;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < threadStackPos && threadIdx.x < SHARED_MEM_SIZE) {
            int globalPos = atomicAdd(d_stackSize, 1);
            d_stack[V + globalPos] = sharedStack[threadIdx.x];
        }
    }
}

// Host function to launch DFS on GPU
void dfsFullyOptimizedCUDA(const std::vector<std::vector<int>>& graph, int start) {
    int V = graph.size();

    // Flatten the graph adjacency matrix
    std::vector<int> flatGraph(V * V, 0);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            flatGraph[i * V + j] = graph[i][j];
        }
    }

    // Device memory allocation
    int* d_graph, * d_stack, * d_stackSize;
    bool* d_visited;
    cudaMalloc((void**)&d_graph, V * V * sizeof(int));
    cudaMalloc((void**)&d_stack, 2 * V * sizeof(int));
    cudaMalloc((void**)&d_visited, V * sizeof(bool));
    cudaMalloc((void**)&d_stackSize, sizeof(int));

    // Host memory allocation
    std::vector<int> h_stack(2 * V, INF);
    std::vector<char> h_visited(V, 0); // Використання std::vector<char> для відвіданих вершин
    int h_stackSize = 0;

    // Initialize the stack with the start vertex
    h_stack[0] = start;
    h_visited[start] = true;
    h_stackSize = 1;

    // Copy data to device
    cudaMemcpy(d_graph, flatGraph.data(), V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stack, h_stack.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited.data(), V * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stackSize, &h_stackSize, sizeof(int), cudaMemcpyHostToDevice);

    // Launch DFS kernel
    int numBlocks = (V + 255) / 256;
    int numThreads = 256;

    while (h_stackSize > 0) {
        cudaMemcpy(d_stackSize, &h_stackSize, sizeof(int), cudaMemcpyHostToDevice);
        dfsKernelFullyOptimized << <numBlocks, numThreads >> > (d_graph, d_stack, d_visited, d_stackSize, V);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_stackSize, d_stackSize, sizeof(int), cudaMemcpyDeviceToHost);

        // Shift the stack
        cudaMemcpy(h_stack.data(), d_stack, 2 * V * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < V; ++i) {
            h_stack[i] = h_stack[V + i];
            h_stack[V + i] = INF;
        }
        cudaMemcpy(d_stack, h_stack.data(), 2 * V * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Copy visited array back to host
    cudaMemcpy(h_visited.data(), d_visited, V * sizeof(char), cudaMemcpyDeviceToHost);

    // Print the DFS result
    //std::cout << "DFS Order: ";
    //for (int i = 0; i < V; ++i) {
    //    if (h_visited[i]) {
    //        std::cout << i << " ";
    //    }
    //}
    //std::cout << std::endl;

    // Free device memory
    cudaFree(d_graph);
    cudaFree(d_stack);
    cudaFree(d_visited);
    cudaFree(d_stackSize);
}