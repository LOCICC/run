#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
namespace gcts{
void linear_assignment_auction(
    const double* cost_matrics,    // 注意：这里拼写疑似应为cost_matrices
    int* solutions,
    const int num_graphs,
    const int num_nodes,
    char* scratch,
    char* stop_flags,
    const double auction_max_eps,
    const double auction_min_eps,
    const double auction_factor,
    const int max_iterations);
void init_auction(
        const int num_graphs, 
        const int num_nodes, 
        char*& scratch, 
        char*& stop_flags
        );
void destroy_auction(
        char* scratch, 
        char* stop_flags
        );
}
