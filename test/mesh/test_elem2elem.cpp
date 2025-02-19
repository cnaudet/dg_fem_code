#include "mesh/Mesh.hpp"
#include <iostream>

int main() {
    // Define a simple 2D mesh (2x2 grid) in a unit square
    int D = 2;
    std::vector<int> num_points = {3, 3};  // 3 points in each dimension
    std::vector<double> lower = {0.0, 0.0};
    std::vector<double> upper = {1.0, 1.0};

    // Create mesh
    Mesh Mesh(D, num_points, lower, upper);

    // Compute face connectivity
    Mesh.compute_face_connectivity();

    // Print connectivity results
    std::cout << "Face connectivity:\n";
    for (const auto& conn : Mesh.get_face_connectivity()) {
        int elem1, face1, elem2, face2;
        std::tie(elem1, face1, elem2, face2) = conn;
        std::cout << "Element " << elem1 << " (face " << face1 << ") <--> "
                  << "Element " << elem2 << " (face " << face2 << ")\n";
    }

    return 0;
}

