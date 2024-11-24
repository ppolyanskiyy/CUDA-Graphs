#ifndef GRAPH_PROCESSING_H
#define GRAPH_PROCESSING_H

#include <vector>
#include <utility>

namespace GraphProcessing {

    /**
     * @brief Enumeration to specify execution mode for algorithms.
     */
    enum class ExecutionMode {
        Sequential,
        ParallelCPU,
        ParallelGPU
    };

    /**
     * @brief Representation of a graph using an adjacency list with edge weights.
     */
    class Graph {
    public:
        /**
         * @brief Constructs a graph with a given number of vertices.
         * @param numVertices Number of vertices in the graph.
         */
        Graph(int numVertices);

        /**
         * @brief Adds an undirected edge between two vertices with a given weight.
         * @param src Source vertex.
         * @param dest Destination vertex.
         * @param weight Weight of the edge (default is 1).
         */
        void addEdge(int src, int dest, int weight = 1);

        /**
         * @brief Adds a directed edge from src to dest with a given weight.
         * @param src Source vertex.
         * @param dest Destination vertex.
         * @param weight Weight of the edge (default is 1).
         */
        void addDirectedEdge(int src, int dest, int weight = 1);

        /**
         * @brief Retrieves the number of vertices in the graph.
         * @return Number of vertices.
         */
        int getNumVertices() const;

        /**
         * @brief Retrieves the adjacency list of the graph.
         * @return Adjacency list.
         */
        const std::vector<std::vector<std::pair<int, int>>>& getAdjacencyList() const;

        /**
         * @brief Performs Breadth-First Search (BFS) on the graph.
         * @param startVertex Starting vertex for BFS.
         * @param mode Execution mode (default is Sequential).
         * @return Vector of vertices in the order they were visited.
         */
        std::vector<int> bfs(int startVertex, ExecutionMode mode = ExecutionMode::Sequential);

        /**
         * @brief Performs Depth-First Search (DFS) on the graph.
         * @param startVertex Starting vertex for DFS.
         * @param mode Execution mode (default is Sequential).
         * @return Vector of vertices in the order they were visited.
         */
        std::vector<int> dfs(int startVertex, ExecutionMode mode = ExecutionMode::Sequential);

        /**
         * @brief Performs Dijkstra's algorithm to find the shortest paths.
         * @param startVertex Starting vertex for Dijkstra's algorithm.
         * @param mode Execution mode (default is Sequential).
         * @return Vector of minimum distances from the start vertex.
         */
        std::vector<int> dijkstra(int startVertex, ExecutionMode mode = ExecutionMode::Sequential);

        /**
         * @brief Performs Floyd–Warshall's algorithm to find the shortest paths.
         * @param mode Execution mode (default is Sequential).
         * @return Vector of minimum distances from the start vertex.
         */
        std::vector<int> floydWarshall(ExecutionMode mode = ExecutionMode::Sequential);

        /**
         * @brief Performs Bellman-Ford algorithm to find shortest paths from a starting vertex.
         * @param startVertex Starting vertex for Bellman-Ford algorithm.
         * @param mode Execution mode (default is Sequential).
         * @return Pair containing a boolean indicating if a negative cycle exists and a vector of distances.
         */
        std::pair<bool, std::vector<int>> bellmanFord(int startVertex, ExecutionMode mode = ExecutionMode::Sequential);

        /**
         * @brief Performs Prim's algorithm to find shortest paths from a starting vertex.
         * @param mode Execution mode (default is Sequential).
         * @return Pair containing a boolean indicating if a negative cycle exists and a vector of distances.
         */
        std::vector<std::pair<int, int>> Graph::prim(ExecutionMode mode = ExecutionMode::Sequential);

    private:

        int numVertices;
        // Each pair consists of (neighbor, weight)
        std::vector<std::vector<std::pair<int, int>>> adjacencyList;
    };

} // namespace GraphProcessing

#endif // GRAPH_PROCESSING_H
