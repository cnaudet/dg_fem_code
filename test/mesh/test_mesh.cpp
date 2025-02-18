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
    cudaError_t err = cudaMemcpy(d_nodes, d_nodes, num_nodes * 2 * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
    }  

    // Print node coordinates
    std::cout << num_nodes << " Nodes:\n";
    // std::cout << d_nodes;
    for (int i = 0; i < num_nodes; ++i) {
        std::cout << h_nodes[i * 2] << " " << h_nodes[i * 2 + 1] << "\n";
    }

    return 0;
}
