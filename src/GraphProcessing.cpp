// GraphProcessing.cpp

#include "GraphProcessing.h"

namespace GraphProcessing {

    // Constructor
    Graph::Graph(int numVertices)
        : numVertices(numVertices), adjacencyList(numVertices) {}

    // Adds an undirected edge between two vertices with a given weight
    void Graph::addEdge(int src, int dest, int weight) {
        adjacencyList[src].emplace_back(dest, weight);
        adjacencyList[dest].emplace_back(src, weight);
    }

    // Adds a directed edge from src to dest with a given weight
    void Graph::addDirectedEdge(int src, int dest, int weight) {
        adjacencyList[src].emplace_back(dest, weight);
    }

    // Retrieves the number of vertices in the graph
    int Graph::getNumVertices() const {
        return numVertices;
    }

    // Retrieves the adjacency list of the graph
    const std::vector<std::vector<std::pair<int, int>>>& Graph::getAdjacencyList() const {
        return adjacencyList;
    }

} // namespace GraphProcessing
