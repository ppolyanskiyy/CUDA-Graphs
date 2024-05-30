#include "GraphTraversal.h"
#include <chrono>

#define MEASURE(func, message, ...) { \
    auto start = std::chrono::high_resolution_clock::now(); \
    func(__VA_ARGS__); \
    auto stop = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start); \
    std::cout << message << ": " << duration.count() << " sec." << std::endl; \
}

std::vector<std::vector<int>> generateRandomGraph(int V, int E) {
    std::vector<std::vector<int>> adj(V);

    srand(time(0));

    for (int i = 0; i < E; ++i) {
        int u = rand() % V;
        int v = rand() % V;
        if (u != v) {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }

    return adj;
}


int main() {
    int V = 1000;
    int E = 5000;

    std::vector<std::vector<int>> graph = generateRandomGraph(V, E);

    MEASURE(bfsSequential, "bfsSequential", graph, 0);
    MEASURE(bfsDefaultCUDA, "bfsDefaultCUDA", graph, 0);
    MEASURE(bfsSharedCUDA, "bfsSharedCUDA", graph, 0);
    MEASURE(bfsCoalescedCUDA, "bfsCoalescedCUDA", graph, 0);
    MEASURE(bfsFullyOptimizedCUDA, "bfsFullyOptimizedCUDA", graph, 0);

    MEASURE(dfsSequential, "dfsSequential", graph, 0);
    MEASURE(dfsDefaultCUDA, "dfsDefaultCUDA", graph, 0);
    MEASURE(dfsSharedCUDA, "dfsSharedCUDA", graph, 0);
    MEASURE(dfsCoalescedCUDA, "dfsCoalescedCUDA", graph, 0);
    MEASURE(dfsFullyOptimizedCUDA, "dfsFullyOptimizedCUDA", graph, 0);

    return 0;
}
