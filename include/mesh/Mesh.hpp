#ifndef MESH_HPP
#define MESH_HPP

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <tuple>
#include <unordered_map>


struct VectorHash {
    size_t operator()(const std::vector<int>& v) const {
        size_t hash = v.size();
        for (int i : v) {
            hash ^= std::hash<int>{}(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};


class Mesh {
public:
    // Constructor: Reads mesh from a file
    Mesh(int D, const std::vector<int>& num_points, const std::vector<double>& lower, const std::vector<double>& upper);
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

    void compute_face_connectivity();
    const std::vector<std::tuple<int, int, int, int>>& get_face_connectivity() const { return face_connectivity; }


    std::vector<int> face2node(size_t elem, size_t face); // Function declaration
private:
    int D;  // Number of dimensions
    size_t num_nodes, num_elements, num_boundaries;

    double* d_nodes;   // Device pointer for node coordinates
    int* d_elements;   // Device pointer for element connectivity
    int* d_boundaries; // Device pointer for boundary information
    std::unordered_map<std::vector<int>, std::tuple<int, int>, VectorHash> face_map;
    std::vector<std::tuple<int,int,int,int>> face_connectivity; // (elem1, face1, elem2, face2)

    // Helper functions
    void generate_nodes(const std::vector<int>& num_points, const std::vector<double>& lower, const std::vector<double>& upper);
    void generate_elements(const std::vector<int>& num_points);
    void generate_boundaries(const std::vector<int>& num_points);
    void identify_faces();
};

#endif // MESH_HPP
