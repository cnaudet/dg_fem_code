cmake_minimum_required(VERSION 3.17)
project(DG_FEM_CODE LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Kokkos (ensure Kokkos is installed or provide its path)
find_package(Kokkos REQUIRED)

# Add include directories
include_directories(include)

# Add source files
add_executable(dg_fem_code src/main.cpp src/mesh/Mesh.cpp)

# Link Kokkos
target_link_libraries(dg_fem_code Kokkos::kokkos)

# Enable testing
enable_testing()
add_subdirectory(test)