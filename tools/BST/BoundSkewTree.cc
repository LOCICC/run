#include "BoundSkewTree.h"
#include "BoundSkewTree3D.h"
#include <chrono>
namespace gcts {

void BoundSkewTree::build(const double skew_bound, const int db_unit, const double unit_h_cap, const double unit_h_res,
  const double unit_v_cap, const double unit_v_res)
{
    g_skew_bound=skew_bound;
    std::cout<<"------ start BoundSkewTree ------\n";
    auto start = std::chrono::high_resolution_clock::now();
    run();
    recursiveBottomUp();
    embedding();
    convert();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "BoundSkewTree execution time: " << elapsed.count() << " seconds" << std::endl;
    std::cout<<"------ end BoundSkewTree ------\n";
}
BoundSkewTree::BoundSkewTree(int num,const std::optional<double>& skew_bound){
   netpinnum=new int[num];
   clusterflag=0;
}
void BoundSkewTree::BoundSkewTreeAddNet(const std::string& net_name, const std::vector<Pin*>& pins){
  for(int i=0;i<pins.size();i++){
    auto* node = new Area(pins[i]->get_location().x(),pins[i]->get_location().y());
  _unmerged_nodes.push_back(node);
  // _node_map.insert({pins[i]->get_name(), pins[i]});
  }
  setPinsize(pins.size());
}
// void BoundSkewTree::setCore(const Point& loc1,const Point& loc2) const
// {
//   // auto* idb_core = _idb_layout->get_core();
//   // auto* core_box = idb_core->get_bounding_box();
//   // auto pt = ctsToIdb(loc);
//   // return core_box->containPoint(pt);
// }
}