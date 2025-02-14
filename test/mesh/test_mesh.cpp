#include "mesh/Mesh.hpp"
#include <iostream>

int main() {
    // Initialize Kokkos
    Kokkos::initialize();

    // Create a mesh object
    Mesh mesh("simple_mesh.txt");

    // Print node coordinates
    auto nodes = mesh.get_nodes();
    std::cout << "Nodes:\n";
    for (size_t i = 0; i < nodes.extent(0); ++i) {
        std::cout << nodes(i, 0) << " " << nodes(i, 1) << "\n";
    }

    // Finalize Kokkos
    Kokkos::finalize();
    return 0;
}