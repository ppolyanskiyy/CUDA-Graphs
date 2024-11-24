cmake_minimum_required(VERSION 3.12)
project(GraphProcessingProject LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support if needed
find_package(CUDA 10.0 REQUIRED)

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)

# Create the static library
add_library(GraphProcessing STATIC ${SRC_FILES})

# Set properties for CUDA files
set_target_properties(GraphProcessing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable position-independent code (PIC) for static library
set_target_properties(GraphProcessing PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Google Test
# Assuming GTest is installed on your system
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Test files
file(GLOB_RECURSE TEST_FILES tests/*.cxx tests/*.cu)

# Add the executable for tests
add_executable(GraphTests ${TEST_FILES})

# Link the test executable with the library and GTest
target_link_libraries(GraphTests
    GraphProcessing
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    pthread
)

# Add CUDA to the test executable if needed
set_target_properties(GraphTests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add tests
add_test(NAME GraphTests COMMAND GraphTests)

# CUDA settings
# Set the compute capability (adjust according to your GPU)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

# If you have multiple CUDA files and want to compile them as C++
set_source_files_properties(${SRC_FILES} ${TEST_FILES} PROPERTIES LANGUAGE CUDA)

