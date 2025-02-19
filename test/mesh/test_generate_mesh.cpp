#include "mesh/Mesh.hpp"
#include <vector>
#include <iostream>

int main() {
    int D = 3; // Example: 3D Mesh
    std::vector<int> num_points = {5, 5, 5};  // 5 points per dimension
    std::vector<double> lower = {0.0, 0.0, 0.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};

    Mesh Mesh(D, num_points, lower, upper);

    std::cout << "Generated Mesh: " << Mesh.get_num_nodes() << " nodes, "
              << Mesh.get_num_elements() << " elements, "
              << Mesh.get_num_boundaries() << " boundary nodes.\n";

    return 0;
}
