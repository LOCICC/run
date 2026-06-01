
#include <string> 
void BoundSkewTreeRun3D(const int net_num, const int* load_pins_num, const int* load_pins_x,const int* load_pins_y, const int* load_pins_z, const double* load_pins_cap, const double skew_bound, 
    int*& root, int*& left_child, int*& right_child, int*& tree_node_x, int*& tree_node_y,int* &tree_node_num);