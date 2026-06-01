/**
 * @file CTS.h
 * @author LOCICC (1150206533@qq.com)
 * @brief GPU-CTS
 * @version 0.1
 * @date 2025-12-15
 * 
 * @copyright Copyright (c) 2025
 * 
 */
// #include "BoundSkewTree.h"
// #include "GPUClustering.h"
// #include "Timing.h"
// #include "CtsCellLib.hh"
// #include "LevelSolve.h"
// #include "TreeMaker.h"
#include <string> 
/**
 * @brief generate multiple bound skew trees and return their structures
 * 
 * @param net_num bound skew trees number
 * @param load_pins_num pin count for each net
 * @param load_pins_x pin coordinates_x array
 * @param load_pins_y pin coordinates_y array
 * @param load_pins_cap pin init cap array
 * @param skew_bound skew
 * @param root  return the root node index, initialized to nullptr
 * @param left_child  return the left child node index array, initialized to nullptr
 * @param right_child  return the left child node index array, initialized to nullptr
 * @param tree_node_x  return the node coordinates_x array, initialized to nullptr
 * @param tree_node_y  return the node coordinates_y array, initialized to nullptr
 */
void BoundSkewTreeRun(const int net_num, const int* load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const double skew_bound, 
    int*& root, int*& left_child, int*& right_child, int*& tree_node_x, int*& tree_node_y,int* &tree_node_num);

/**
 * @brief generate multiple CBSTrees and return their structures
 * 
 * @param net_num CBS trees number
 * @param load_pins_num pin count for each net
 * @param load_pins_x pin coordinates_x array
 * @param load_pins_y pin coordinates_y array
 * @param load_pins_cap pin init cap array
 * @param skew_bound skew
 * @param root  return the root node index, initialized to nullptr
 * @param left_child  return the left child node index array, initialized to nullptr
 * @param right_child  return the left child node index array, initialized to nullptr
 * @param tree_node_x  return the node coordinates_x array, initialized to nullptr
 * @param tree_node_y  return the node coordinates_y array, initialized to nullptr
 */
void CBSTreeRun(const int net_num, const int* load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const double skew_bound,
    int* &root, int* &left_child, int* &right_child, int* &tree_node_x, int* &tree_node_y,int* &tree_node_num);

/**
 * @brief pin node clustering
 * 
 * @param load_pins_num Total number of pins 
 * @param load_pins_x pin coordinates_x array
 * @param load_pins_y pin coordinates_y array
 * @param load_pins_cap pin init cap array
 * @param max_fanout 
 * @param iters 
 * @param no_change_stop 
 * @param cluster_num return the number of clusters
 * @param clusters_segment return pin count per cluster in clustering
 * @param load_pins_index return pins index id
 */
void IterClusteringRun(const int load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const size_t& max_fanout,
                           const size_t& iters, const size_t& no_change_stop,int &cluster_num, int *&clusters_segment, int *&load_pins_index);

/**
 * @brief Enhanced pin node clustering
 * 
 * @param load_pins_num total number of pins 
 * @param load_pins_x pin coordinates_x array
 * @param load_pins_y pin coordinates_y array
 * @param load_pins_cap pin init cap array
 * @param max_fanout 
 * @param cluster_num previous number of clusters
 * @param clusters_segment pin count per cluster in previous clustering
 * @param max_len 
 * @param iters 
 * @param next_cluster_num return the number of clusters
 * @param next_clusters_segment return pin count per cluster in current clustering
 * @param next_load_pins_index return pins index id
 */
void EnhanceClusteringRun(const int load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const size_t max_fanout,
                          const int cluster_num, const int *clusters_segment, const int max_len, const size_t& iters, 
                          int &next_cluster_num, int *&next_clusters_segment, int *&next_load_pins_index);

/**
 * @brief single level clock tree synthesis
 * 
 * @param load_pins_num total number of pins 
 * @param load_pins_x pin coordinates_x array
 * @param load_pins_y pin coordinates_y array
 * @param load_pins_cap pin init cap array
 * @param max_fanout 
 * @param max_len 
 * @param fixed_num 
 * @param fixed_x
 * @param fixed_y
 * @param next_cluster_num return the number of clusters
 * @param next_clusters_segment return pin count per cluster in current clustering
 * @param next_load_pins_index return pins index id
 * @param next_level_load_pins_num return number of pin nodes in the next layer
 * @param next_level_load_pins_x return coordinates_x of pin nodes in the next layer
 * @param next_level_load_pins_y return coordinates_y of pin nodes in the next layer
 * @param next_level_load_pins_cap return coordinates_cap of pin nodes in the next layer
 * @param buffer_cell_masters return buffer type in the layer
 */
void LevelRun(const int load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const int max_fanout, const double max_len,
    const int fixed_num, const int* fixed_x,const int* fixed_y,
    int &next_cluster_num, int *&next_clusters_segment, int *&next_load_pins_index,
    int &next_level_load_pins_num, int *&next_level_load_pins_x, int *&next_level_load_pins_y, double *&next_level_load_pins_cap, std::string *&buffer_cell_masters);
/**
 * @brief 
 * 
 * @param load_pins_num 
 * @param load_pins_x 
 * @param load_pins_y 
 * @param load_pins_cap 
 * @param max_fanout 
 * @param max_len 
 * @param level 
 */
void MultiLevelRun(const int load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const int max_fanout, const double max_len, 
    const int fixed_num, const int* fixed_x,const int* fixed_y,int &root_x,int &root_y, std::string &root_cell_master, int &level);
/**
 * @brief 
 * 
 * @param load_pins_num total number of pins 
 * @param load_pins_x pin coordinates_x array
 * @param load_pins_y pin coordinates_y array
 * @param load_pins_cap pin init cap array
 * @param skew_bound skew
 * @param root  return the root node index, initialized to nullptr
 * @param left_child  return the left child node index array, initialized to nullptr
 * @param right_child  return the left child node index array, initialized to nullptr
 * @param tree_node_x  return the node coordinates_x array, initialized to nullptr
 * @param tree_node_y  return the node coordinates_y array, initialized to nullptr
 */
void BoundSkewTreeSingleRun(const int load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const double skew_bound, 
    int& root, int*& left_child, int*& right_child, int*& tree_node_x, int*& tree_node_y, int &tree_node_num);

/**
 * @brief 
 * 
 * @param load_pins_num total number of pins 
 * @param load_pins_x pin coordinates_x array
 * @param load_pins_y pin coordinates_y array
 * @param load_pins_cap pin init cap array
 * @param skew_bound skew
 * @param root  return the root node index, initialized to nullptr
 * @param left_child  return the left child node index array, initialized to nullptr
 * @param right_child  return the left child node index array, initialized to nullptr
 * @param tree_node_x  return the node coordinates_x array, initialized to nullptr
 * @param tree_node_y  return the node coordinates_y array, initialized to nullptr
 */
void CBSTreeSingleRun(const int load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const double skew_bound, const int type,
    int& root,  int*& left_child, int*& right_child, int* &tree_node_x, int* & tree_node_y,int &tree_node_num);

/**
 * @brief initializing RC values
 * 
 * @param unit_h_cap /pF
 * @param unit_h_res /kohm
 * @param unit_v_cap /pF
 * @param unit_v_res /kohm
 * @param db_unit 
 */
void TimingInit(const double unit_h_cap, const double unit_h_res, const double unit_v_cap, const double unit_v_res, const int db_unit);
/**
 * @brief initializing Cell values
 * 
 * @param cell_master 
 * @param slew_in_index_num 
 * @param slew_in_index 
 * @param cap_out_index_num 
 * @param cap_out_index 
 * @param delay_mid_value_num 
 * @param delay_mid_value 
 * @param slew_mid_value_num 
 * @param slew_mid_value 
 */
void AddCell(const std::string& cell_master,  const int slew_in_index_num, const double* slew_in_index, const int cap_out_index_num, 
    const double* cap_out_index, const int delay_mid_value_num, const double* delay_mid_value, const int slew_mid_value_num, const double* slew_mid_value);

/**
 * @brief initializing Cell init cap
 * 
 * @param cell_master 
 * @param init_cap 
 */
void AddCellCap(const std::string& cell_master, const double init_cap);

/**
 * @brief initializing buffer types
 * 
 * @param buffer_num 
 * @param cell_masters 
 */
void AddBuffers(int buffer_num, const std::string* cell_masters);
void AddBuffers(const std::string& cell_master, const int slew_in_index_num, const double* slew_in_index,
    const int cap_out_index_num, const double* cap_out_index, const int delay_mid_value_num, const double* delay_mid_value,
    const int slew_mid_value_num, const double* slew_mid_value,const double init_cap);

void SkewRun(const int net_num, const int* load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const double skew_bound, 
    int*& root, int*& left_child, int*& right_child, int* &tree_node_x, int* &tree_node_y,int* &tree_node_num);