#include "mesh/Mesh.hpp"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

int main() {
    // Create a mesh object
    Mesh mesh("simple_mesh.txt");

    // Get node data from device
    double* d_nodes = mesh.get_nodes();
    int num_nodes = mesh.get_num_nodes();

    // Allocate host memory
    std::vector<double> h_nodes(num_nodes * 2);
    
    // Copy data from device to host
    cudaMemcpy(h_nodes.data(), d_nodes, num_nodes * 2 * sizeof(double), cudaMemcpyDeviceToHost);

    // Print node coordinates
    std::cout << "Nodes:\n";
    for (int i = 0; i < num_nodes; ++i) {
        std::cout << h_nodes[i * 2] << " " << h_nodes[i * 2 + 1] << "\n";
    }

    return 0;
}
