#include "mesh/Mesh.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

// Constructor: Reads mesh from a file
Mesh::Mesh(const std::string& filename) {
    d_nodes = nullptr;
    d_elements = nullptr;
    d_boundaries = nullptr;
    read_custom_mesh(filename);
}

// Destructor: Free allocated memory
Mesh::~Mesh() {
    cudaFree(d_nodes);
    cudaFree(d_elements);
    cudaFree(d_boundaries);
}

// Helper function to read a simple custom mesh format
void Mesh::read_custom_mesh(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open mesh file: " + filename);
    }

    std::vector<double> temp_nodes;
    std::vector<int> temp_elements;
    std::vector<int> temp_boundaries;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "Node") {
            double x, y;
            iss >> x >> y;
            temp_nodes.push_back(x);
            temp_nodes.push_back(y);
        } else if (token == "Element") {
            int n1, n2, n3;
            iss >> n1 >> n2 >> n3;
            temp_elements.push_back(n1);
            temp_elements.push_back(n2);
            temp_elements.push_back(n3);
        } else if (token == "Boundary") {
            int b1, b2;
            iss >> b1 >> b2;
            temp_boundaries.push_back(b1);
            temp_boundaries.push_back(b2);
        }
    }

    num_nodes = temp_nodes.size()/2;
    num_elements = temp_elements.size()/3;
    num_boundaries = temp_boundaries.size()/2;

    std::cout << "Read " << num_nodes << " nodes:" << std::endl;
    for (size_t i = 0; i < temp_nodes.size(); i += 2) {
        std::cout << temp_nodes[i] << " " << temp_nodes[i + 1] << std::endl;
    }

    cudaMalloc(&d_nodes, num_nodes* 2 * sizeof(double));
    cudaMalloc(&d_elements, num_elements * 3 * sizeof(int));
    cudaMalloc(&d_boundaries, num_boundaries * 2 * sizeof(int));

    cudaMemcpy(d_nodes, temp_nodes.data(), num_nodes * 2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elements, temp_elements.data(), num_elements * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boundaries, temp_boundaries.data(), num_boundaries * 2 * sizeof(int), cudaMemcpyHostToDevice);
}
