#include "mesh/Mesh.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <unordered_map>



Mesh::Mesh(int D, const std::vector<int>& num_points, const std::vector<double>& lower, const std::vector<double>& upper)
    : D(D), num_nodes(1), num_elements(1), num_boundaries(0) {

    for (int d = 0; d < D; ++d)
        num_nodes *= num_points[d];

    num_elements = 1;
    for (int p : num_points) {
        num_elements *= (p - 1);
    }

    generate_nodes(num_points, lower, upper);
    generate_elements(num_points);
    generate_boundaries(num_points);
}

Mesh::~Mesh() {
    cudaFree(d_nodes);
    cudaFree(d_elements);
    cudaFree(d_boundaries);
}

// Generate node coordinates
void Mesh::generate_nodes(const std::vector<int>& num_points, const std::vector<double>& lower, const std::vector<double>& upper) {
    std::vector<double> h_nodes(num_nodes * D);
    
    std::vector<double> dx(D);
    for (int d = 0; d < D; ++d)
        dx[d] = (upper[d] - lower[d]) / (num_points[d] - 1);

    for (size_t n = 0; n < num_nodes; ++n) {
        size_t index = n;
        for (int d = D - 1; d >= 0; --d) {
            int coord_index = index % num_points[d];
            h_nodes[n * D + d] = lower[d] + coord_index * dx[d];
            index /= num_points[d];
        }
    }

    cudaMalloc(&d_nodes, num_nodes * D * sizeof(double));
    cudaMemcpy(d_nodes, h_nodes.data(), num_nodes * D * sizeof(double), cudaMemcpyHostToDevice);
}

// Generate hypercube elements
void Mesh::generate_elements(const std::vector<int>& num_points) {
    int num_neighbors = std::pow(2, D);  // 2^D vertices per hypercube
    std::vector<int> h_elements;

    for (size_t n = 0; n < num_nodes; ++n) {
        size_t index = n;
        std::vector<int> base_index(D);
        bool is_valid = true;

        for (int d = D - 1; d >= 0; --d) {
            base_index[d] = index % num_points[d];
            if (base_index[d] >= num_points[d] - 1) is_valid = false;
            index /= num_points[d];
        }

        if (!is_valid) continue;

        std::vector<int> element(num_neighbors);
        for (int i = 0; i < num_neighbors; ++i) {
            int offset = 0;
            for (int d = 0; d < D; ++d) {
                if (i & (1 << d)) offset += std::pow(num_points[d], d);
            }
            element[i] = n + offset;
        }

        h_elements.insert(h_elements.end(), element.begin(), element.end());
    }

    num_elements = h_elements.size() / num_neighbors;
    cudaMalloc(&d_elements, h_elements.size() * sizeof(int));
    cudaMemcpy(d_elements, h_elements.data(), h_elements.size() * sizeof(int), cudaMemcpyHostToDevice);
}

// Identify boundary nodes
void Mesh::generate_boundaries(const std::vector<int>& num_points) {
    std::vector<int> h_boundaries;
    
    for (size_t n = 0; n < num_nodes; ++n) {
        size_t index = n;
        bool is_boundary = false;

        for (int d = D - 1; d >= 0; --d) {
            int coord_index = index % num_points[d];
            if (coord_index == 0 || coord_index == num_points[d] - 1)
                is_boundary = true;
            index /= num_points[d];
        }

        if (is_boundary)
            h_boundaries.push_back(n);
    }

    num_boundaries = h_boundaries.size();
    cudaMalloc(&d_boundaries, num_boundaries * sizeof(int));
    cudaMemcpy(d_boundaries, h_boundaries.data(), num_boundaries * sizeof(int), cudaMemcpyHostToDevice);
}




void Mesh::compute_face_connectivity() {
    if (!d_elements) {
        std::cerr << "Error: d_elements is not initialized!" << std::endl;
        return;
    }
    face_map.clear();
    face_connectivity.clear();

    for (size_t elem = 0; elem < num_elements; ++elem) {
        for (size_t face = 0; face < 2 * D; ++face) { // 2 faces per dimension
            std::vector<int> face_nodes;

            for (size_t node = 0; node < num_nodes; ++node) {
                if (/* Condition that determines nodes on this face */) {
                    face_nodes.push_back(d_elements[elem * num_nodes + node]);
                }
            }

            std::sort(face_nodes.begin(), face_nodes.end());

            auto it = face_map.find(face_nodes);
            if (it == face_map.end()) {
                face_map[face_nodes] = std::make_tuple(elem, face);
            } else {
                auto face_tuple = it->second;
                face_connectivity.push_back(std::make_tuple(std::get<0>(face_tuple), std::get<1>(face_tuple), elem, face));
                face_map.erase(it);
            }
        }
    }
}


std::vector<int> Mesh::face2node(size_t elem, size_t face) {
    std::vector<int> face_nodes;

    if (D == 2) { // For 2D case (quadrilateral element)
        const in nnpe = 4
        if (face == 0) { // Top face
            face_nodes = {elem * nnpe + 0, elem * nnpe + 1};
        } else if (face == 1) { // Right face
            face_nodes = {elem * nnpe + 0, elem * nnpe + 2};
        } else if (face == 2) { // Bottom face
            face_nodes = {elem * nnpe + 1, elem * nnpe + 3};
        } else if (face == 3) { // Left face
            face_nodes = {elem * nnpe + 2, elem * nnpe + 3};
        }
    } else if (D == 3) { // For 3D case (hexahedral element)
        const int nnpe = 8
        if (face == 0) { // Bottom face
            face_nodes = {elem * nnpe + 0, elem * nnpe + 1, elem * nnpe + 2, elem * nnpe + 3};
        } else if (face == 1) { // Top face
            face_nodes = {elem * nnpe + 4, elem * nnpe + 5, elem * nnpe + 6, elem * nnpe + 7};
        } else if (face == 2) { // Front face
            face_nodes = {elem * nnpe + 0, elem * nnpe + 1, elem * nnpe + 4, elem * nnpe + 5};
        } else if (face == 3) { // Back face
            face_nodes = {elem * nnpe + 2, elem * nnpe + 0, elem * nnpe + 6, elem * nnpe + 4};
        } else if (face == 4) { // Left face
            face_nodes = {elem * nnpe + 1, elem * nnpe + 3, elem * nnpe + 5, elem * nnpe + 7};
        } else if (face == 5) { // Right face
            face_nodes = {elem * nnpe + 3, elem * nnpe + 2, elem * nnpe + 7, elem * nnpe + 6};
        }
    }

    return face_nodes;
}
