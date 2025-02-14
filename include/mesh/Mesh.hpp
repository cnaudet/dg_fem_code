#ifndef MESH_HPP
#define MESH_HPP

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <iostream>

class Mesh {
public:
    // Constructor: Reads mesh from a file
    Mesh(const std::string& filename);
    ~Mesh();

    // Get node coordinates (N_nodes x dim)
    double* get_nodes() const { return d_nodes; }
    
    // Get element connectivity (N_elements x N_nodes_per_element)
    int* get_elements() const { return d_elements; }
    
    // Get boundary information (N_boundaries x ...)
    int* get_boundaries() const { return d_boundaries; }

private:
    double* d_nodes;   // Device pointer for node coordinates
    int* d_elements;   // Device pointer for element connectivity
    int* d_boundaries; // Device pointer for boundary information

    size_t num_nodes;
    size_t num_elements;
    size_t num_boundaries;

    // Helper function to read a simple custom mesh format
    void read_custom_mesh(const std::string& filename);
};

#endif // MESH_HPP