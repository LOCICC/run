#include "Timing.h"
#include "Pingpu.hh"
#include "Instgpu.hh"
#include "Nodegpu.hh"
#include "Enumgpu.hh"
#include "CtsCellLib.hh"
namespace gcts {
int Timing::_db_unit = 0;
double Timing::_unit_cap = 0.0;  // pf
double Timing::_unit_res = 0.0;  // kilo-ohm
double Timing::_unit_h_cap = 0.0;
double Timing::_unit_h_res = 0.0;
double Timing::_unit_v_cap = 0.0;
double Timing::_unit_v_res = 0.0;
std::vector<std::string> Timing::bufferlibs;
std::vector<std::string> Timing::Cobufferlibs;
std::unordered_map<std::string, CtsCellLib*> CtsLibs::_lib_maps;
  void Timing::update(std::vector<Node*>nets,std::vector<Inst*>& insts,std::vector<Node*> load_nodes,int num)
  {
    netLenPropagate(nets,num);
    capPropagate(nets,insts,load_nodes,num);
    // slewPropagate(insts,nets,num);
    // cellDelayPropagate(insts,load_nodes,num);
    // wireDelayPropagate(nets,num);
  }
  void Timing::checkskew(std::vector<Node*>nets,std::vector<Inst*>& insts,std::vector<Node*> load_nodes,int num){
    capPropagate(nets,insts,load_nodes,num);
    // std::cout<<nets[0]->get_cap_load()<<'\n';
    slewPropagate(nets,num);
    // std::cout<<"slew over\n";
    cellDelayPropagate(nets,num);
    // std::cout<<"cellDelay over\n";
    wireDelayPropagate(nets,num);
    std::cout<<nets[0]->get_min_delay()<<' '<<nets[0]->get_max_delay()<<'\n';
  }
  void Timing::netLenPropagate(std::vector<Node*>nets,int num)
  {
    for(int i=0;i<num;i++)nets[i]->postOrder(updateNetLen);
  }
  void Timing::wireDelayPropagate(std::vector<Node*>nets,int num)
  {
    for(int i=0;i<num;i++)nets[i]->postOrder(updateWireDelay);
  }
  void Timing::capPropagate(std::vector<Node*>nets,std::vector<Inst*>& insts,std::vector<Node*> load_nodes,int num)
  {
    for(int i=0;i<load_nodes.size();i++){
      // std::cout<<load_nodes[i]->_cap_load<<'\n';
      // load_nodes[i]->_cap_load=insts[load_nodes[i]->instID]->cap;
    }
    // std::cout<<"capPropagate\n";
    for(int i=0;i<num;i++)nets[i]->postOrder(updateCapLoad);
  }
  void Timing::resetNet(Node* driver_node){
    driver_node->postOrder([&](Node* node) {
      node->_parent = nullptr;
      node->_children.clear();
    });
  }
  void Timing::cellDelayPropagate(std::vector<Node*>nets ,int num)
  {
    for(int i=0;i<num;i++){
      updateCellDelay(nets[i]);
    }
  }
  void Timing::updateCellDelay(Node* load_node){
      // if (!inst->isBuffer()) {
      //   return;
      // }
      auto slew_in = load_node->get_slew_in();
      auto cap_load = load_node->get_cap_load();
      auto cell_name = load_node->get_cell_master();
      if (cell_name.empty()) { 
        return;
      }
      if (cap_load == 0) {
        // inst->set_insert_delay(0);
        load_node->set_min_delay(0);
        load_node->set_max_delay(0);
        return;
      }
      // auto* lib = CTSAPIInst.getCellLib(cell_name);
      // std::cout<<"updateCellDelay "<<slew_in<<' '<<cap_load<<'\n';
      auto insert_delay = CtsLibs::_lib_maps[cell_name]->calcDelay(slew_in, cap_load);
      // inst->set_insert_delay(insert_delay);
      auto min_delay = load_node->get_min_delay();
      auto max_delay = load_node->get_max_delay();
      load_node->set_min_delay(insert_delay + min_delay);
      load_node->set_max_delay(insert_delay + max_delay);
      // std::cout<<"updateCellDelay "<<insert_delay<<' '<<slew_in<<' '<<cap_load<<'\n';
  }
  void Timing::slewPropagate(std::vector<Node*>nets,int num)
  {
    for(int i=0;i<num;i++){
      auto cell_name = nets[i]->get_cell_master();
      if(cell_name.empty())continue;
      // std::cout<<i<<' '<<cell_name<<' '<<nets[i]->get_cap_load()<<'\n';
      auto slew_out = CtsLibs::_lib_maps[cell_name]->calcSlew(nets[i]->get_cap_load());
      // std::cout<<i<<' '<<cell_name<<' '<<nets[i]->get_cap_load()<<' '<<slew_out<<"\n";
      nets[i]->set_slew_in(slew_out);
    }
    for(int i=0;i<num;i++)nets[i]->preOrder(updateSlewIn);
  }
  
  double Timing::calcElmoreDelay(const double& cap, const double& len)
  {
    return _unit_res * len * (_unit_cap * len / 2 + cap);
  }
  double Timing::calcElmoreDelay(const double& cap, const double& x, const double& y, const RCPattern& pattern)
  {
    double delay = 0;
    switch (pattern) {
    case RCPattern::kSingle:
      delay = calcElmoreDelay(cap, x + y);
      break;
    case RCPattern::kHV:
      delay = _unit_h_res * x * (_unit_h_cap * x / 2 + cap) + _unit_v_res * y * (_unit_v_cap * y / 2 + cap + _unit_h_cap * x);
      break;
    case RCPattern::kVH:
      delay = _unit_v_res * y * (_unit_v_cap * y / 2 + cap) + _unit_h_res * x * (_unit_h_cap * x / 2 + cap + _unit_v_cap * y);
      break;
    default:
      break;
    }
    return delay;
  }
  void Timing::init(){
    _unit_cap=2.90402e-06;
    _unit_res=0.00075;
    _unit_h_cap=1.94215e-06;
    _unit_h_res=0.00178571;
    _unit_v_cap=2.90402e-06;
    _unit_v_res=0.00075;
    _db_unit=2000;
  }
  void Timing::init(const double unit_h_cap, const double unit_h_res, const double unit_v_cap, const double unit_v_res, const int db_unit)
  {
    // set RC, db unit and liberty info from api
    _unit_cap=unit_h_cap;
    _unit_res=unit_h_cap;
    _unit_h_cap=unit_h_cap;
    _unit_h_res=unit_h_res;
    _unit_v_cap=unit_v_cap;
    _unit_v_res=unit_v_res;
    _db_unit=db_unit;

    // _unit_cap=7.595e-05;
    // _unit_res=7.83333e-05;
    // _unit_h_cap=8.57718e-05;
    // _unit_h_res=7.83333e-05;
    // _unit_v_cap=7.595e-05;
    // _unit_v_res=7.83333e-05;
    // _db_unit=1000;

    // _unit_cap=2.90402e-06;
    // _unit_res=0.00075;
    // _unit_h_cap=1.94215e-06;
    // _unit_h_res=0.00178571;
    // _unit_v_cap=2.90402e-06;
    // _unit_v_res=0.00075;
    // _delay_libs = CTSAPIInst.getAllBufferLibs();
    // _root_lib = CTSAPIInst.getRootBufferLib();
    // // set algorithm parameters from config
    // auto* config = CTSAPIInst.get_config();
    // _skew_bound = config->get_skew_bound();
    // _max_buf_tran = config->get_max_buf_tran();
    // _max_sink_tran = config->get_max_sink_tran();
    // _max_cap = config->get_max_cap();
    // _max_fanout = config->get_max_fanout();
    // _min_length = config->get_min_length();
    // _max_length = config->get_max_length();
    // // temp para
    // _min_insert_delay = _delay_libs.front()->getDelayIntercept();
  }
  void Timing::AddCellLib(const std::string& cell_master, const int slew_in_index_num, const double* slew_in_index,
    const int cap_out_index_num, const double* cap_out_index, const int delay_mid_value_num, const double* delay_mid_value,
    const int slew_mid_value_num, const double* slew_mid_value, const double init_cap){ 
    std::vector<std::vector<double>> index_list;
    index_list.reserve(2);  

    index_list.emplace_back(slew_in_index, slew_in_index + slew_in_index_num);
    index_list.emplace_back(cap_out_index, cap_out_index + cap_out_index_num);
    std::vector<double> delay_values(delay_mid_value, 
                                    delay_mid_value + delay_mid_value_num);
    std::vector<double> slew_values(slew_mid_value, 
                                   slew_mid_value + slew_mid_value_num);
    auto lib = new CtsCellLib(cell_master, index_list, delay_values, slew_values);
    lib->set_init_cap(init_cap);
    CtsLibs::_lib_maps[cell_master]=lib;
    
    auto slew_in = index_list[0];
    auto cap_out = index_list[1];
    std::vector<double> x_slew_in;
    std::vector<double> x_cap_out;
    std::vector<double> y_delay;
    std::vector<double> y_slew;

    for (size_t i = 0; i < slew_in.size(); ++i) {
        auto work_slew = slew_in[i];
        for (size_t j = 0; j < cap_out.size(); ++j) {
        auto work_cap = cap_out[j];
        x_slew_in.emplace_back(work_slew);
        x_cap_out.emplace_back(work_cap);
        y_delay.emplace_back(delay_mid_value[i * cap_out.size() + j]);
        y_slew.emplace_back(slew_mid_value[i * cap_out.size() + j]);
        }
    }
    auto _model_factory = new ModelFactory();
    std::vector<std::vector<double>> x_delay = {x_slew_in, x_cap_out};
    lib->set_delay_coef(_model_factory->cppLinearModel(x_delay, y_delay));

    std::vector<std::vector<double>> x_slew = {x_cap_out};
    lib->set_slew_coef(_model_factory->cppLinearModel(x_slew, y_slew));
  }

  void Timing::AddCellLib(const std::string& cell_master, const double init_cap){ 
    auto lib = new CtsCellLib(cell_master, init_cap);
    lib->set_init_cap(init_cap);
    CtsLibs::_lib_maps[cell_master]=lib;
  }
}