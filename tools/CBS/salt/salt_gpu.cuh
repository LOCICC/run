#pragma once

#include <cassert>
#include <iostream>
#include <map>
#include <vector>

#ifdef SALT_BUILD_DLL
#define SALT_API __host__ __attribute__((visibility("default")))
#else
#define SALT_API __host__ __attribute__((visibility("default")))
#endif

namespace salt::cuda
{
/// @brief data type
typedef unsigned long long DTYPE;

/// @brief GPU Kernel THREADS PER BLOCK
constexpr unsigned int BLOCK = 256;

/// @brief small nodes use a n^3 algorithm, default 45
constexpr unsigned int SALT_SAMLL_NET = 20;

inline
    /**
     * @brief total number of blocks for x threads
     *
     * @param x total threads
     * @return total number of blocks
     */
    unsigned int
    BLOCKS(int x)
{
    return (x + BLOCK - 1) / BLOCK;
}

/**
 * @brief GPU-accelerated Salt algorithm
 *
 * @param[in] num_nets Total number of nets
 * @param[in] net_acc FLUTE accuracy parameter for each net
 * @param[in] skip_net Boolean array indicating whether to skip a net:
 *            - true = the corresponding net does not require processing
 *            - false = the net needs to be processed (size: num_nets)
 * @param[in] net_start Start index array for each net
 * @param[in] net_x X-coordinate array of original net nodes
 * @param[in] net_y Y-coordinate array of original net nodes
 * @param[in] epsilon Trade-off parameter for the Salt algorithm
 * @param[out] salt_tree_start Start index array for each Salt tree
 * @param[out] salt_x X-coordinate array of output Salt tree nodes
 * @param[out] salt_y Y-coordinate array of output Salt tree nodes
 * @param[out] salt_n Local parent ID array for each Salt tree node
 *             - salt_n[i] = i: The node is the root of the tree
 *             - salt_n[i] = 0xFFFFFFFF (4294967295): The node is invalid/unused
 */
SALT_API void salt_cuda(uintptr_t num_nets, uint32_t *net_acc,
                        const bool *skip_net, uintptr_t *net_start,
                        DTYPE *net_x, DTYPE *net_y, double epsilon,
                        uintptr_t *salt_tree_start, DTYPE *salt_x,
                        DTYPE *salt_y, uint32_t *salt_n);

/**
 * @brief CUDA-accelerated FLUTE algorithm for Steiner tree generation
 *
 * @param[in] num_nets Total number of nets to process
 * @param[in] net_acc FLUTE accuracy parameter for each net
 * @param[in] skip_net Boolean array indicating skip status for each net:
 *            - true = skip processing the net
 *            - false = process the net normally
 * @param[in] net_start Start index array for each net's nodes
 * @param[in] net_x X-coordinate array of original net pin positions
 * @param[in] net_y Y-coordinate array of original net pin positions
 * @param[out] result_x X-coordinate array of FLUTE-generated Steiner tree nodes
 * @param[out] result_y Y-coordinate array of FLUTE-generated Steiner tree nodes
 * @param[out] result_n Local parent ID array for Steiner tree nodes
 * @param[out] result_x_orig_id Original index mapping for X-coordinates
 * @param[out] result_y_orig_id Original index mapping for Y-coordinates
 * @param[in] lut_path File path to the FLUTE lookup table (LUT)
 *
 * @note 2*d slots are allocated per net (d = net degree), but only 2*d-2 are
 * used (last 2 slots are unused/blank)
 */
SALT_API void flute_cuda(uintptr_t num_nets, uint32_t *net_acc,
                         const bool *skip_net, uintptr_t *net_start,
                         DTYPE *net_x, DTYPE *net_y, DTYPE *flute_x,
                         DTYPE *flute_y, uint32_t *flute_n,
                         uint32_t *flute_x_orig_id, uint32_t *flute_y_orig_id,
                         const char *lut_path);

/**
 * @brief CUDA-accelerated Salt refinement algorithm for optimizing existing
 * Steiner trees
 *
 * @param[in] num_nets Total number of nets
 * @param[in] net_start Start index array for each net's pin positions
 * @param[in] net_x X-coordinate array of original net pin positions
 * @param[in] net_y Y-coordinate array of original net pin positions
 * @param[in] tree_start Start index array for each input Steiner tree's nodes
 * @param[in] roots Global root ID array for each input tree
 * @param[in] tree_x X-coordinate array of input tree nodes
 * @param[in] tree_y Y-coordinate array of input tree nodes
 * @param[in] tree_n Local parent/neighbor ID array for input tree nodes
 * @param[in] epsilon Salt algorithm trade-off parameter
 * @param[out] salt_tree_start Start index array for each refined Salt tree
 * @param[out] salt_x X-coordinate array of refined Salt tree nodes
 * @param[out] salt_y Y-coordinate array of refined Salt tree nodes
 * @param[out] salt_n Local parent ID array for refined Salt tree nodes
 *             - salt_n[i] = i: The node is the root of the refined tree
 *             - salt_n[i] = 0xFFFFFFFF (4294967295): The node is invalid/unused
 */
SALT_API void salt_refine_cuda(uintptr_t num_nets, uintptr_t *net_start,
                               DTYPE *net_x, DTYPE *net_y,
                               uintptr_t *tree_start, uintptr_t *roots,
                               DTYPE *tree_x, DTYPE *tree_y, uint32_t *tree_n,
                               double epsilon, uintptr_t *salt_tree_start,
                               DTYPE *salt_x, DTYPE *salt_y, uint32_t *salt_n);

/**
 * @brief CUDA-accelerated breakpoint detection for Steiner tree pins
 *
 * @param[in] num_nets Total number of nets
 * @param[in] net_start Start index array for each net's pin positions
 * @param[in] net_x X-coordinate array of original net pin positions
 * @param[in] net_y Y-coordinate array of original net pin positions
 * @param[in] tree_start Start index array for each input Steiner tree's nodes
 * @param[in] roots Global root ID array for each input tree
 * @param[in] tree_x X-coordinate array of input tree nodes
 * @param[in] tree_y Y-coordinate array of input tree nodes
 * @param[in] tree_n Local parent/neighbor ID array for input tree nodes
 * @param[in] epsilon Salt algorithm trade-off parameter
 * @param[out] result_n Updated local parent ID array for tree nodes, array
 *      Special value: 0xFFFFFFFF (4294967295) = the node is invalid/unused
 * @param[out] breakpoint Boolean array marking breakpoint pins:
 *             - true = the corresponding pin is a breakpoint
 *             - false = the pin is not a breakpoint
 */
SALT_API void breakpoint_detection_cuda(intptr_t num_nets, uintptr_t *net_start,
                                        DTYPE *net_x, DTYPE *net_y,
                                        uintptr_t *tree_start, uintptr_t *roots,
                                        DTYPE *tree_x, DTYPE *tree_y,
                                        uint32_t *tree_n, double epsilon,
                                        uint32_t *result_n, bool *breakpoint);

/**
 * @brief CUDA-accelerated RSMA (Rectilinear Steiner Minimum Arborescence)
 * generation for Steiner trees
 *
 * @param[in] num_nets Total number of nets
 * @param[in] tree_start Start index array for each input Steiner tree's nodes
 * @param[in] roots Global root ID array for each input tree
 * @param[in] tree_x X-coordinate array of input tree nodes
 * @param[in] tree_y Y-coordinate array of input tree nodes
 * @param[in] tree_n Local parent/neighbor ID array for input tree nodes
 * @param[in] rsma_nodes Boolean array marking nodes for RSMA generation:
 *            - true = generate RSMA with this node
 *            - false = skip RSMA for this node (root is false)
 * @param[out] result_tree_start Start index array for each generated RSMA tree
 * @param[out] result_x X-coordinate array of RSMA tree nodes
 * @param[out] result_y Y-coordinate array of RSMA tree nodes
 * @param[out] result_n Local parent ID array for RSMA tree nodes
 *             - result_n[i] = i: The node is the root of the RSMA tree
 *             - result_n[i] = 0xFFFFFFFF (4294967295): The node is
 * invalid/unused
 */
SALT_API void
rsma_generation_cuda(intptr_t num_nets, uintptr_t *tree_start, uintptr_t *roots,
                     DTYPE *tree_x, DTYPE *tree_y, uint32_t *tree_n,
                     bool *rsma_nodes, uintptr_t *result_tree_start,
                     DTYPE *result_x, DTYPE *result_y, uint32_t *result_n);
} // namespace salt::cuda