
#pragma once
#include <iostream> 
#include "Nodegpu.hh"
#include "Instgpu.hh"
#include "Pingpu.hh"
#include <cmath>
namespace gcts {

enum class LayerPattern
{
  kH,  // horizontal
  kV,  // vertical
  kNone,
};
class Timing
{
 public:
  Timing() = delete;
  ~Timing() = default;

  static void init();
  static void init(const double unit_h_cap, const double unit_h_res, const double unit_v_cap, const double unit_v_res, const int db_unit);
  // net based
  static void updatePinCap(Pin* pin);
  static void resetNet(Node* driver_node);
  static void update(std::vector<Node*>nets,std::vector<Inst*>& insts,std::vector<Node*> load_nodes,int num);
  static void checkskew(std::vector<Node*>nets,std::vector<Inst*>& insts,std::vector<Node*> load_nodes,int num);
  static void netLenPropagate(std::vector<Node*>nets,int num);
  static void capPropagate(std::vector<Node*>nets,std::vector<Inst*>& insts,std::vector<Node*> load_nodes,int num);
  static void slewPropagate(std::vector<Node*>nets,int num);
  static void cellDelayPropagate(std::vector<Node*>nets,int num);
  static void wireDelayPropagate(std::vector<Node*>nets,int num);
  // inst based
  static void updateCellDelay(Node* load_node);
  static double calcSkew(Node* node);
  static bool skewFeasible(Node* node, const std::optional<double>& skew_bound = std::nullopt);
  // pin based
  static void initLoadPinDelay(Pin* pin, const bool& by_cell = false);
  // timing calc
  static double calcElmoreDelay(const double& cap, const double& len);
  static double calcElmoreDelay(const double& cap, const double& x, const double& y, const RCPattern& pattern = RCPattern::kSingle);
  // info getter
  static double getUnitCap(const LayerPattern& pattern = LayerPattern::kNone);
  static double getUnitRes(const LayerPattern& pattern = LayerPattern::kNone);
  // get
  static double getSkewBound() { return _skew_bound; }
  static int getDbUnit() { return _db_unit; }
  static double getMaxBufTran() { return _max_buf_tran; }
  static double getMaxSinkTran() { return _max_sink_tran; }
  static double getMaxCap() { return _max_cap; }
  static int getMaxFanout() { return _max_fanout; }
  static double getMinLength() { return _min_length; }
  static double getMaxLength() { return _max_length; }
  static double getMinInsertDelay() { return _min_insert_delay; }
  static void AddCellLib(const std::string& cell_master, const int slew_in_index_num, const double* slew_in_index,
    const int cap_out_index_num, const double* cap_out_index, const int delay_mid_value_num, const double* delay_mid_value,
    const int slew_mid_value_num, const double* slew_mid_value, const double init_cap);
  static void AddCellLib(const std::string& cell_master, const double init_cap);
  // static icts::CtsCellLib* getMinSizeLib() { return _delay_libs.front(); }
  // static icts::CtsCellLib* getMaxSizeLib() { return _delay_libs.back(); }
  // static icts::CtsCellLib* getRootSizeLib() { return _root_lib; }
  // static std::string getMinSizeCell() { return getMinSizeLib()->get_cell_master(); }
  // static std::string getMaxSizeCell() { return getMaxSizeLib()->get_cell_master(); }
  // static std::string getRootSizeCell() { return getRootSizeLib()->get_cell_master(); }
  // static std::vector<icts::CtsCellLib*> getDelayLibs() { return _delay_libs; }
  // node based
  static void updateNetLen(Node* node)
  {
    auto net_len = calcNetLen(node);
    node->set_sub_len(net_len);
  }
  static void updateCapLoad(Node* node, const RCPattern& pattern = RCPattern::kSingle)
  {
    auto cap_load = calcCapLoad(node, pattern);
    node->set_cap_load(cap_load);
  }

  static void updateSlewIn(Node* node, const RCPattern& pattern = RCPattern::kSingle)
  {
    auto calc_slew_in = [&node](Node* child) {
      auto slew_ideal = calcIdealSlew(node, child);
      double slew_in = std::sqrt(std::pow(node->get_slew_in(), 2) + std::pow(slew_ideal, 2));
      child->set_slew_in(slew_in);
    };
    std::for_each(node->get_children().begin(), node->get_children().end(), calc_slew_in);
  }

  static void updateWireDelay(Node* node, const RCPattern& pattern = RCPattern::kSingle)
  {
    if (node->get_children().empty()) {
      return;
    }
    double min_delay = std::numeric_limits<double>::max();
    double max_delay = std::numeric_limits<double>::min();
    auto calc_delay = [&min_delay, &max_delay, &node, &pattern](Node* child) {
      auto delay = calcElmoreDelay(node, child, pattern);
      min_delay = std::min(min_delay, delay + child->get_min_delay());
      max_delay = std::max(max_delay, delay + child->get_max_delay());
    };
    std::for_each(node->get_children().begin(), node->get_children().end(), calc_delay);
    if (node->isLoad()) {
        min_delay = std::min(min_delay, node->get_min_delay());
        max_delay = std::max(max_delay, node->get_max_delay());
    }
    node->set_min_delay(min_delay);
    node->set_max_delay(max_delay);
  }

  static double calcNetLen(Node* node)
  {
    double net_len = 0;
    auto accumulate_net_len = [&net_len, &node](Node* child) {
      auto nodept=node->get_location();
      auto childpt=child->get_location();
      net_len += calcLen(node, child) + child->get_sub_len();
      net_len += child->get_required_snake();
    };
    std::for_each(node->get_children().begin(), node->get_children().end(), accumulate_net_len);
    return net_len;
  }

  static double calcCapLoad(Node* node, const RCPattern& pattern = RCPattern::kSingle)
  {
    double cap_load = 0;
    // std::cout<<node->get_location().x()<<' '<<node->get_location().y()<<' '<<node->isLoad()<<' '<<node->get_children().size()<<'\n';
    if (node->isLoad()) {
      cap_load = node->get_cap_load();
    }
    auto accumulate_cap = [&cap_load, &node, &pattern](Node* child) {
      // switch (pattern) {
      //   // normal rc pattern
      //   case RCPattern::kSingle:
      //     cap_load += _unit_cap * calcLen(node, child, LayerPattern::kNone) + child->get_cap_load();
      //     cap_load += _unit_h_cap * child->get_required_snake();
      //     break;
      //   // HV or VH pattern
      //   default:
      //     cap_load += _unit_h_cap * calcLen(node, child, LayerPattern::kH) + _unit_v_cap * calcLen(node, child, LayerPattern::kV)
      //                 + child->get_cap_load();
      //     cap_load += _unit_h_cap * child->get_required_snake();
      //     break;
      // }
      cap_load += _unit_cap * calcLen(node, child, LayerPattern::kNone)/_db_unit + child->get_cap_load();
      cap_load += _unit_h_cap * child->get_required_snake();
      // std::cout<<_unit_cap<<' '<<calcLen(node, child, LayerPattern::kNone)<<' '<<cap_load<<'\n';
    };
    std::for_each(node->get_children().begin(), node->get_children().end(), accumulate_cap);
    return cap_load;
  }

  static double calcIdealSlew(Node* parent, Node* child, const RCPattern& pattern = RCPattern::kSingle)
  {
    return std::log(9) * calcElmoreDelay(parent, child, pattern);
  }
  static double calcElmoreDelay(Node* parent, Node* child, const RCPattern& pattern = RCPattern::kSingle)
  {
    double delay = 0;
    auto cap_load = child->get_cap_load();
    switch (pattern) {
      // normal rc pattern
      case RCPattern::kSingle: {
        auto len = calcLen(parent, child);
        delay = calcElmoreDelay(cap_load, len + child->get_required_snake());
        break;
      }
      // HV or VH pattern
      default: {
        auto x = calcLen(parent, child, LayerPattern::kH);
        auto y = calcLen(parent, child, LayerPattern::kV);
        delay = calcElmoreDelay(cap_load, x, y, pattern);
        delay += calcElmoreDelay(cap_load + _unit_h_cap * x + _unit_v_cap * y, child->get_required_snake(), 0, pattern);
        break;
      }
    }
    return delay;
  }


  static double calcLen(Node* parent, Node* child, const LayerPattern& pattern = LayerPattern::kNone)
  {
    return calcDist(parent->get_location(), child->get_location(), pattern);
  }
  static double calcLinearSlew(const double& cap_out);
  static double calcDist(const Point& p1, const Point& p2, const LayerPattern& pattern = LayerPattern::kNone)
  {
    double dist = 0;
    switch (pattern) {
      case LayerPattern::kNone:
        dist = std::fabs(p1.x() - p2.x()) + std::fabs(p1.y() - p2.y());
        break;
      case LayerPattern::kH:
        dist = std::fabs(p1.x() - p2.x());
        break;
      case LayerPattern::kV:
        dist = std::fabs(p1.y() - p2.y());
        break;
      default:
        break;
    }
    return dist;
  }

  double kEpsilon = 1e-6;

  static double _unit_cap;  // pf
  static double _unit_res;  // kilo-ohm
  static double _unit_h_cap;
  static double _unit_h_res;
  static double _unit_v_cap;
  static double _unit_v_res;
  static double _skew_bound;
  static int _db_unit;
  static double _max_buf_tran;
  static double _max_sink_tran;
  static double _max_cap;
  static int _max_fanout;
  static double _min_length;
  static double _max_length;
  static double _min_insert_delay;
  // static std::vector<gcts::CtsCellLib*> _delay_libs;
  // static icts::CtsCellLib* _root_lib;
  static std::vector<std::string> bufferlibs;
  static std::vector<std::string> Cobufferlibs;
};

class CellLib{

  public:
  // double calcDelay(const double& slew_in, const double& cap_out) const
  // {
  //   return calcInsertDelay(slew_in, cap_out);
  // }
  // double calcInsertDelay(const double& slew_in, const double& cap_out) const
  // {
  //   // slew_in index
  //   auto it1 = std::upper_bound(_slew_in_span.begin(), _slew_in_span.end(), slew_in);
  //   size_t i1 = std::clamp(static_cast<size_t>(std::distance(_slew_in_span.begin(), it1)), size_t{1}, _slew_in_span.size() - 1);
  //   double slew_low = _slew_in_span[i1 - 1];
  //   double slew_high = _slew_in_span[i1];

  //   // cap_out index
  //   auto it2 = std::upper_bound(_cap_out_span.begin(), _cap_out_span.end(), cap_out);
  //   size_t i2 = std::clamp(static_cast<size_t>(std::distance(_cap_out_span.begin(), it2)), size_t{1}, _cap_out_span.size() - 1);
  //   double cap_low = _cap_out_span[i2 - 1];
  //   double cap_high = _cap_out_span[i2];

  //   double y11 = _delay_values[(i1 - 1) * _cap_out_span.size() + i2 - 1];
  //   double y12 = _delay_values[(i1 - 1) * _cap_out_span.size() + i2];
  //   double y21 = _delay_values[i1 * _cap_out_span.size() + i2 - 1];
  //   double y22 = _delay_values[i1 * _cap_out_span.size() + i2];

  //   // insert value
  //   double res = std::lerp(std::lerp(y11, y12, std::lerp(0.0, 1.0, (cap_out - cap_low) / (cap_high - cap_low))),
  //                          std::lerp(y21, y22, std::lerp(0.0, 1.0, (cap_out - cap_low) / (cap_high - cap_low))),
  //                          std::lerp(0.0, 1.0, (slew_in - slew_low) / (slew_high - slew_low)));

  //   return res;
  // }
  // double calcLinearSlew(const double& cap_out) const { return _slew_coef[0] + _slew_coef[1] * cap_out; }
  // double calcSlew(const double& cap_out) const
  // {
  //   return calcLinearSlew(cap_out);
  // }
  // std::span<const double> _slew_in_span;
  // std::span<const double> _cap_out_span;
};

// static std::unordered_map<std::string, CellLib*> Lib_map;
}  // namespace icts