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
    
    // Get information
    size_t get_num_nodes() const { return num_nodes; }
    size_t get_num_elements() const { return num_elements; }
    size_t get_num_boundaries() const { return num_boundaries; }
private:
    int D;  // Number of dimensions
    size_t num_nodes, num_elements, num_boundaries;

    double* d_nodes;   // Device pointer for node coordinates
    int* d_elements;   // Device pointer for element connectivity
    int* d_boundaries; // Device pointer for boundary information

    size_t num_nodes;
    size_t num_elements;
    size_t num_boundaries;

    // Helper function to read a simple custom mesh format
    void read_custom_mesh(const std::string& filename);

    // Helper functions
    void generate_nodes(const std::vector<int>& num_points, const std::vector<double>& lower, const std::vector<double>& upper);
    void generate_elements(const std::vector<int>& num_points);
    void generate_boundaries(const std::vector<int>& num_points);
};

#endif // MESH_HPP
