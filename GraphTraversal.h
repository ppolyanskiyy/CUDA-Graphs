#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <queue>
#include <stack>

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#define INF 10000000
#define SHARED_MEM_SIZE 256


void bfsSequential(const std::vector<std::vector<int>> & graph, int start);
void bfsDefaultCUDA(const std::vector<std::vector<int>> & graph, int start);
void bfsSharedCUDA(const std::vector<std::vector<int>> & graph, int start);
void bfsCoalescedCUDA(const std::vector<std::vector<int>> & graph, int start);
void bfsFullyOptimizedCUDA(const std::vector<std::vector<int>> & graph, int start);


void dfsSequential(const std::vector<std::vector<int>> & graph, int start);
void dfsDefaultCUDA(const std::vector<std::vector<int>> & graph, int start);
void dfsSharedCUDA(const std::vector<std::vector<int>> & graph, int start);
void dfsCoalescedCUDA(const std::vector<std::vector<int>> & graph, int start);
void dfsFullyOptimizedCUDA(const std::vector<std::vector<int>> & graph, int start);
