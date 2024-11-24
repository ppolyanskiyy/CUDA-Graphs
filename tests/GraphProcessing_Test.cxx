// GraphTests.cpp

#include "GraphProcessing.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <limits>

using namespace GraphProcessing;

// Helper function to compare two vectors regardless of order
template <typename T>
bool compareVectorsUnordered(const std::vector<T>& v1, const std::vector<T>& v2) {
    std::vector<T> sorted_v1 = v1;
    std::vector<T> sorted_v2 = v2;
    std::sort(sorted_v1.begin(), sorted_v1.end());
    std::sort(sorted_v2.begin(), sorted_v2.end());
    return sorted_v1 == sorted_v2;
}

// Helper function to check if two 2D vectors are equal
template <typename T>
bool compare2DVectors(const std::vector<std::vector<T>>& v1, const std::vector<std::vector<T>>& v2) {
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (v1[i].size() != v2[i].size()) return false;
        for (size_t j = 0; j < v1[i].size(); ++j) {
            if (v1[i][j] != v2[i][j]) return false;
        }
    }
    return true;
}

// Test fixture class for Graph tests
class GraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up a sample graph for testing
        // This graph will have 6 vertices numbered from 0 to 5
        graph = new Graph(6);

        // Add edges
        graph->addEdge(0, 1, 7);
        graph->addEdge(0, 2, 9);
        graph->addEdge(0, 5, 14);
        graph->addEdge(1, 2, 10);
        graph->addEdge(1, 3, 15);
        graph->addEdge(2, 3, 11);
        graph->addEdge(2, 5, 2);
        graph->addEdge(3, 4, 6);
        graph->addEdge(4, 5, 9);
    }

    void TearDown() override {
        delete graph;
    }

    Graph* graph;
};

TEST_F(GraphTest, GraphConstruction) {
    // Test the number of vertices
    EXPECT_EQ(graph->getNumVertices(), 6);

    // Test the adjacency list size
    const auto& adjacencyList = graph->getAdjacencyList();
    EXPECT_EQ(adjacencyList.size(), 6);

    // Test edges
    EXPECT_FALSE(adjacencyList[0].empty());
    EXPECT_FALSE(adjacencyList[1].empty());
    EXPECT_FALSE(adjacencyList[2].empty());
    EXPECT_FALSE(adjacencyList[3].empty());
    EXPECT_FALSE(adjacencyList[4].empty());
    EXPECT_FALSE(adjacencyList[5].empty());
}

TEST_F(GraphTest, BFSSequential) {
    std::vector<int> expectedOrder = {0, 1, 2, 5, 3, 4};
    std::vector<int> bfsResult = graph->bfs(0, ExecutionMode::Sequential);
    EXPECT_EQ(bfsResult, expectedOrder);
}

TEST_F(GraphTest, BFSParallelCPU) {
    std::vector<int> expectedOrder = {0, 1, 2, 5, 3, 4};
    std::vector<int> bfsResult = graph->bfs(0, ExecutionMode::ParallelCPU);
    EXPECT_EQ(bfsResult, expectedOrder);
}

TEST_F(GraphTest, BFSParallelGPU) {
    std::vector<int> expectedOrder = {0, 1, 2, 5, 3, 4};
    std::vector<int> bfsResult = graph->bfs(0, ExecutionMode::ParallelGPU);
    EXPECT_EQ(bfsResult, expectedOrder);
}

TEST_F(GraphTest, DFSSequential) {
    std::vector<int> expectedOrder = {0, 1, 2, 5, 3, 4};
    std::vector<int> dfsResult = graph->dfs(0, ExecutionMode::Sequential);
    EXPECT_EQ(dfsResult, expectedOrder);
}

TEST_F(GraphTest, DFSParallelCPU) {
    // Due to the nature of parallel DFS, the visit order may differ
    std::vector<int> dfsResult = graph->dfs(0, ExecutionMode::ParallelCPU);

    // Check that all vertices are visited
    std::vector<int> expectedVertices = {0, 1, 2, 3, 4, 5};
    EXPECT_TRUE(compareVectorsUnordered(dfsResult, expectedVertices));
    EXPECT_EQ(dfsResult.size(), expectedVertices.size());
}

TEST_F(GraphTest, DFSParallelGPU) {
    // Due to the nature of parallel DFS, the visit order may differ
    std::vector<int> dfsResult = graph->dfs(0, ExecutionMode::ParallelGPU);

    // Check that all vertices are visited
    std::vector<int> expectedVertices = {0, 1, 2, 3, 4, 5};
    EXPECT_TRUE(compareVectorsUnordered(dfsResult, expectedVertices));
    EXPECT_EQ(dfsResult.size(), expectedVertices.size());
}

TEST_F(GraphTest, DijkstraSequential) {
    std::vector<int> expectedDistances = {0, 7, 9, 20, 20, 11};
    std::vector<int> distances = graph->dijkstra(0, ExecutionMode::Sequential);
    EXPECT_EQ(distances, expectedDistances);
}

TEST_F(GraphTest, DijkstraParallelCPU) {
    std::vector<int> expectedDistances = {0, 7, 9, 20, 20, 11};
    std::vector<int> distances = graph->dijkstra(0, ExecutionMode::ParallelCPU);
    EXPECT_EQ(distances, expectedDistances);
}

TEST_F(GraphTest, DijkstraParallelGPU) {
    std::vector<int> expectedDistances = {0, 7, 9, 20, 20, 11};
    std::vector<int> distances = graph->dijkstra(0, ExecutionMode::ParallelGPU);
    EXPECT_EQ(distances, expectedDistances);
}

TEST_F(GraphTest, FloydWarshallSequential) {
    const int INF = std::numeric_limits<int>::max();
    std::vector<std::vector<int>> expectedDistances = {
        {0, 7, 9, 20, 26, 11},
        {7, 0, 10, 15, 21, 12},
        {9, 10, 0, 11, 17, 2},
        {20, 15, 11, 0, 6, 13},
        {26, 21, 17, 6, 0, 9},
        {11, 12, 2, 13, 9, 0}
    };

    auto distances = graph->floydWarshall(ExecutionMode::Sequential);
    EXPECT_TRUE(compare2DVectors(distances, expectedDistances));
}

TEST_F(GraphTest, FloydWarshallParallelCPU) {
    const int INF = std::numeric_limits<int>::max();
    std::vector<std::vector<int>> expectedDistances = {
        {0, 7, 9, 20, 26, 11},
        {7, 0, 10, 15, 21, 12},
        {9, 10, 0, 11, 17, 2},
        {20, 15, 11, 0, 6, 13},
        {26, 21, 17, 6, 0, 9},
        {11, 12, 2, 13, 9, 0}
    };

    auto distances = graph->floydWarshall(ExecutionMode::ParallelCPU);
    EXPECT_TRUE(compare2DVectors(distances, expectedDistances));
}

TEST_F(GraphTest, FloydWarshallParallelGPU) {
    const int INF = std::numeric_limits<int>::max();
    std::vector<std::vector<int>> expectedDistances = {
        {0, 7, 9, 20, 26, 11},
        {7, 0, 10, 15, 21, 12},
        {9, 10, 0, 11, 17, 2},
        {20, 15, 11, 0, 6, 13},
        {26, 21, 17, 6, 0, 9},
        {11, 12, 2, 13, 9, 0}
    };

    auto distances = graph->floydWarshall(ExecutionMode::ParallelGPU);
    EXPECT_TRUE(compare2DVectors(distances, expectedDistances));
}

TEST_F(GraphTest, BellmanFordSequential) {
    // For Bellman-Ford, create a graph with a negative weight edge
    Graph bfGraph(5);
    bfGraph.addDirectedEdge(0, 1, -1);
    bfGraph.addDirectedEdge(0, 2, 4);
    bfGraph.addDirectedEdge(1, 2, 3);
    bfGraph.addDirectedEdge(1, 3, 2);
    bfGraph.addDirectedEdge(1, 4, 2);
    bfGraph.addDirectedEdge(3, 2, 5);
    bfGraph.addDirectedEdge(3, 1, 1);
    bfGraph.addDirectedEdge(4, 3, -3);

    int startVertex = 0;
    auto [success, distances] = bfGraph.bellmanFord(startVertex, ExecutionMode::Sequential);
    EXPECT_TRUE(success);

    std::vector<int> expectedDistances = {0, -1, 2, -2, 1};
    EXPECT_EQ(distances, expectedDistances);
}

TEST_F(GraphTest, BellmanFordParallelCPU) {
    // For Bellman-Ford, create a graph with a negative weight edge
    Graph bfGraph(5);
    bfGraph.addDirectedEdge(0, 1, -1);
    bfGraph.addDirectedEdge(0, 2, 4);
    bfGraph.addDirectedEdge(1, 2, 3);
    bfGraph.addDirectedEdge(1, 3, 2);
    bfGraph.addDirectedEdge(1, 4, 2);
    bfGraph.addDirectedEdge(3, 2, 5);
    bfGraph.addDirectedEdge(3, 1, 1);
    bfGraph.addDirectedEdge(4, 3, -3);

    int startVertex = 0;
    auto [success, distances] = bfGraph.bellmanFord(startVertex, ExecutionMode::ParallelCPU);
    EXPECT_TRUE(success);

    std::vector<int> expectedDistances = {0, -1, 2, -2, 1};
    EXPECT_EQ(distances, expectedDistances);
}

TEST_F(GraphTest, BellmanFordParallelGPU) {
    // For Bellman-Ford, create a graph with a negative weight edge
    Graph bfGraph(5);
    bfGraph.addDirectedEdge(0, 1, -1);
    bfGraph.addDirectedEdge(0, 2, 4);
    bfGraph.addDirectedEdge(1, 2, 3);
    bfGraph.addDirectedEdge(1, 3, 2);
    bfGraph.addDirectedEdge(1, 4, 2);
    bfGraph.addDirectedEdge(3, 2, 5);
    bfGraph.addDirectedEdge(3, 1, 1);
    bfGraph.addDirectedEdge(4, 3, -3);

    int startVertex = 0;
    auto [success, distances] = bfGraph.bellmanFord(startVertex, ExecutionMode::ParallelGPU);
    EXPECT_TRUE(success);

    std::vector<int> expectedDistances = {0, -1, 2, -2, 1};
    EXPECT_EQ(distances, expectedDistances);
}

TEST_F(GraphTest, BellmanFordNegativeCycle) {
    // Create a graph with a negative weight cycle
    Graph bfGraph(3);
    bfGraph.addDirectedEdge(0, 1, 1);
    bfGraph.addDirectedEdge(1, 2, -1);
    bfGraph.addDirectedEdge(2, 0, -1);

    int startVertex = 0;
    auto [success, distances] = bfGraph.bellmanFord(startVertex, ExecutionMode::Sequential);
    EXPECT_FALSE(success);
}

TEST_F(GraphTest, PrimSequential) {
    auto mstEdges = graph->prim(ExecutionMode::Sequential);

    // Since MST can have multiple valid solutions, we check the total weight
    int totalWeight = 0;
    for (const auto& edge : mstEdges) {
        int u = edge.first;
        int v = edge.second;
        // Find the weight of the edge
        int weight = -1;
        for (const auto& neighbor : graph->getAdjacencyList()[u]) {
            if (neighbor.first == v) {
                weight = neighbor.second;
                break;
            }
        }
        EXPECT_NE(weight, -1); // Edge should exist
        totalWeight += weight;
    }

    EXPECT_EQ(totalWeight, 37); // The total weight of the MST for this graph
}

TEST_F(GraphTest, PrimParallelCPU) {
    auto mstEdges = graph->prim(ExecutionMode::ParallelCPU);

    // Since MST can have multiple valid solutions, we check the total weight
    int totalWeight = 0;
    for (const auto& edge : mstEdges) {
        int u = edge.first;
        int v = edge.second;
        // Find the weight of the edge
        int weight = -1;
        for (const auto& neighbor : graph->getAdjacencyList()[u]) {
            if (neighbor.first == v) {
                weight = neighbor.second;
                break;
            }
        }
        EXPECT_NE(weight, -1); // Edge should exist
        totalWeight += weight;
    }

    EXPECT_EQ(totalWeight, 37); // The total weight of the MST for this graph
}

TEST_F(GraphTest, PrimParallelGPU) {
    auto mstEdges = graph->prim(ExecutionMode::ParallelGPU);

    // Since MST can have multiple valid solutions, we check the total weight
    int totalWeight = 0;
    for (const auto& edge : mstEdges) {
        int u = edge.first;
        int v = edge.second;
        // Find the weight of the edge
        int weight = -1;
        for (const auto& neighbor : graph->getAdjacencyList()[u]) {
            if (neighbor.first == v) {
                weight = neighbor.second;
                break;
            }
        }
        EXPECT_NE(weight, -1); // Edge should exist
        totalWeight += weight;
    }

    EXPECT_EQ(totalWeight, 37); // The total weight of the MST for this graph
}

