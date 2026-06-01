#include <stack>
#include <queue>
#include <cub/cub.cuh>
#include "BoundSkewTree.h"
#include "Instgpu.hh"
#include "Nodegpu.hh"
#include "Timing.h"
namespace gcts {
__global__ void initMrAndConvstep(Area* areas,Pt* pts1,Pt* pts2,int beginID,int end){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if((id+beginID)>=end)return ;
  areas[id+beginID].convex_hull.pts=&pts1[32*id];
  areas[id+beginID].convex_hull.size=0;
  areas[id+beginID].mr.pts=&pts2[32*id];
  areas[id+beginID].mr.size=0;
  areas[id+beginID].location=Pt(areas[id+beginID].x,areas[id+beginID].y);
  if(areas[id+beginID].l==areas[id+beginID].r){
    areas[id+beginID].convex_hull.size=1;
    areas[id+beginID].convex_hull.pts[0]=Pt(areas[areas[id+beginID].l].x,areas[areas[id+beginID].l].y);
    areas[id+beginID].mr.size=1;
    areas[id+beginID].mr.pts[0]=Pt(areas[areas[id+beginID].l].x,areas[areas[id+beginID].l].y);
    areas[id+beginID].cap=areas[areas[id+beginID].l].cap;
    // areas[id+beginID].location=Pt(areas[id+beginID].x,areas[id+beginID].y);
  }
  // printf("%d %d %d %d %d %f\n",id+beginID,areas[id+beginID].left,areas[id+beginID].right,areas[id+beginID].l,areas[id+beginID].r,areas[id+beginID].cap);
}
Inst* genBufInst(const std::string& prefix, const Point& location)
{
  auto buf_name = prefix + "_buf";
  auto buf_inst = new Inst(buf_name, location, InstType::kBuffer);
  return buf_inst;
}
void BoundSkewTree::GPUtoCPU(Pointc* pts){
    int areasize=levelsum[maxlevel];
    Area* h_areas = new Area[areasize];
    Pointc* h_pts = new Pointc[pinsize];
    cudaMemcpy(h_areas,areas,areasize*sizeof(Area),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pts,pts,pinsize*sizeof(Pointc),cudaMemcpyDeviceToHost);
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<pinsize;i++){
      auto loc = Point(h_areas[i].x*Timing::_db_unit, h_areas[i].y*Timing::_db_unit);  
      // std::cout<<loc.x()<<' '<<loc.y()<<'\n';
      Node * node = new Node(std::to_string(i), loc);
      node->set_type(NodeType::kSinkPin);
      node->_cap_load=h_areas[i].cap;
      _node_map.insert({h_areas[i].p, node});
      // std::cout<<h_areas[i].p<<' ';
      node->instID=h_pts[i].instID;
      load_nodes.push_back(node);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Salt GPUtoCPU calculate time: " << elapsed.count() << " seconds" << std::endl;
    // std::cout << "pinsize:"<<pinsize<<'\n';
    for(int i=0;i<net_num;i++){
      SaltConvert(&h_areas[pinsize+i],h_areas,i);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - end;
    // std::cout << "SaltConvert calculate time: " << elapsed1.count() << " seconds" << std::endl;
     delete[]h_areas;
     delete[]h_pts;
}
double calcSkew(Node* node){
    auto min_delay = node->get_min_delay();
    auto max_delay = node->get_max_delay();
    return max_delay - min_delay;
}
bool skewFeasible(Node* node, double skew_bound){
    auto skew = calcSkew(node);
    auto delta = skew - skew_bound;
    if (delta > 0 && delta < kEpsilon) {
      node->set_min_delay(node->get_max_delay() - skew_bound);
      return true;
    }
    return skew <= skew_bound;
}
void initLoadPinDelay(Node* pin, const bool& by_cell){
      pin->set_min_delay(0);
      pin->set_max_delay(0);
}
void BoundSkewTree::setArealr(int id){
  if(H_areas[id].left!=-1){
    setArealr(H_areas[id].left);
    H_areas[id].l=H_areas[H_areas[id].left].l;
  }
  if(H_areas[id].right!=-1){
    setArealr(H_areas[id].right);
    H_areas[id].r=H_areas[H_areas[id].right].r;
  }
  if(H_areas[id].left==-1&&H_areas[id].right==-1){
    H_areas[pinid].x=H_areas[id].x,H_areas[pinid].y=H_areas[id].y,H_areas[pinid].cap=H_areas[id].cap;
    H_areas[pinid].p=id;
    H_areas[id].l=H_areas[id].r=pinid;

    int xx=H_areas[pinid].x*Timing::_db_unit,yy=H_areas[pinid].y*Timing::_db_unit;
    std::string s=std::to_string(xx)+std::to_string(yy);
    H_areas[pinid].ptsid=map[s];
    pinid++;
  }
  H_areas[id].location=Pt(H_areas[id].x,H_areas[id].y);
  H_areas[id].snake=0;
  // std::cout<<id<<' '<<H_areas[id].left<<' '<<H_areas[id].right<<' '<<H_areas[id].l<<' '<<H_areas[id].r<<'\n';
}
void BoundSkewTree::initMrAndConv(int cnt){
  Pt* pts1,*pts2;
  cudaMalloc(&pts1, 32*(cnt-pinsize) * sizeof(Pt));
  cudaMalloc(&pts2, 32*(cnt-pinsize) * sizeof(Pt));
  int blockSize=256;
  int gridSize = (cnt-pinsize + blockSize - 1) / blockSize;
  initMrAndConvstep<<<gridSize, blockSize>>>(areas,pts1,pts2,pinsize,cnt);
  cudaDeviceSynchronize();
}
void BoundSkewTree::CPUtoGPU(Pointc* pts){
  std::queue<Node*>nodes[2];
  std::queue<int>nodesid[2];
  int level=0,cnt=0,flag=0;

  Pointc* h_pts=new Pointc[pinsize];
  // std::cout<<pinsize<<"\n";
  cudaMemcpy(h_pts,pts,pinsize*sizeof(Pointc),cudaMemcpyDeviceToHost);
  for(int i=0;i<pinsize;i++){
    int xx=h_pts[i].x*Timing::_db_unit,yy=h_pts[i].y*Timing::_db_unit;
    std::string s=std::to_string(xx)+std::to_string(yy);
    map[s]=i;
  }
  // for(int i=1730;i<1740;i++)std::cout<<h_pts[i].x<<' '<<h_pts[i].y<<'\n';

  for(int i=0;i<pinsize;i++){
    Area area;
    area.id=cnt++;
    // area.ptsid=h_pts[i].instID;
    H_areas.push_back(area);
  }
  for(int i=0;i<net_num;i++){
    nodes[0].push(_root_buf_node[i]);
    nodesid[0].push(cnt);
    Area area(_root_buf_node[i]->get_location().x(),_root_buf_node[i]->get_location().y());
    area.p=-1,area.id=cnt++;
    H_areas.push_back(area);
  }
  levelsum[level]=pinsize;
  levelsum[++level]=cnt;
  int pinid=0;
  while(nodes[flag].size()>0){
    // std::cout<<"level:"<<level<<" nodesize:"<<nodes[flag].size()<<'\n';
    while(nodes[flag].size()>0){
      auto* node=nodes[flag].front();
      int id=nodesid[flag].front();
      H_areas[id].id=id;
      if(node->get_children().size()==0&&node->get_id()>0){
        H_areas[id].ptsid=node->get_id()-1;
        
      }
      for(int i=0;i<node->get_children().size();i++){
        Node *child=node->get_children()[i];
        nodesid[!flag].push(cnt);
        nodes[!flag].push(child);
        Area area(child->get_location().x()*1.0/Timing::_db_unit,child->get_location().y()*1.0/Timing::_db_unit,child->_cap_load);
        area.p=id;
        if(i==0)H_areas[id].left=cnt++;
        else H_areas[id].right=cnt++;
        H_areas.push_back(area);
      }
      nodes[flag].pop();
      nodesid[flag].pop();
      // std::cout<<"childrensize:"<<id<<" "<<H_areas[id].left<<' '<<H_areas[id].right<<'\n';
    }
    if(nodes[!flag].size()>0)levelsum[++level]=cnt;
    flag=!flag;
  }
  maxlevel=level;
  for(int i=0;i<net_num;i++)setArealr(pinsize+i);
  cudaMemcpy(areas,H_areas.data(),cnt*sizeof(Area),cudaMemcpyHostToDevice);
  // coutAreas(areas,levelsum[maxlevel-1],levelsum[maxlevel]-1);
  initMrAndConv(cnt);
}
void BoundSkewTree::runTwo(int id,bool estimation,double _skew_bound){
  // Copy topology
  _root_buf_node[id]->postOrder([&](Node* node) {
    if (node->isPin() && node->isLoad()) {
      if (estimation) {
        initLoadPinDelay(node, true);
      } 
      // TimingPropagator::updatePinCap(pin);
      if (!skewFeasible(node, _skew_bound)) {
        node->set_min_delay(node->get_max_delay() - _skew_bound);
      }
    }
  });
}
double distanceB(const Pt& p1, const Pt& p2)
{
  return std::abs(p1.x - p2.x) + std::abs(p1.y - p2.y);
}
void BoundSkewTree::SaltConvert(Area* _root,Area* h_areas,int net_id){
  std::stack<Area*> stack;
  stack.push(_root);
  // pre-order build Node, leaf node will in _node_map
  while (!stack.empty()) {
    auto* cur = stack.top();
    stack.pop();
    // if(cur->right==0)break;
    if (cur->right!=-1) {
      // std::cout<<"cur->right "<<cur->right<<'\n';
      stack.push(&h_areas[cur->right]);
    }
    if (cur->left!=-1) {
      // std::cout<<"cur->left "<<cur->left<<'\n';
      stack.push(&h_areas[cur->left]);
    }
    if(cur->id<pinsize)continue;
    auto pt = cur->location;
    auto loc = Point(pt.x*1.*Timing::_db_unit, pt.y*1.*Timing::_db_unit);   
    // std::cout<<loc.x()<<' '<<loc.y()<<'\n';
    if (cur->p==-1) {
      // std::cout<<"net_id "<<net_id<<'\n';
      // is root, make buffer
      _root_buf.push_back(genBufInst(std::to_string(net_id), loc));
      // if(pt.x>799&&pt.x<800&&pt.y>295&&pt.y<296)std::cout<<"SaltConvert "<<pt.x<<' '<<pt.y<<'\n';
      auto* node=new Node(_root_buf[net_id]->get_driver_pin()->get_name(),loc);
      node->set_type(NodeType::kBufferPin);
      _root_buf_node.push_back(node);
      _node_map.insert({cur->id, node});
      continue;
    }
    auto* parent = &h_areas[cur->p];
    Node* node = nullptr;
    Node* parent_node = _node_map[cur->p]; 
    // std::cout<<"cur->id "<<cur->id<<' '<<cur->p<<' '<<cur->left<<' '<<cur->right<<'\n';
    if (cur->left == -1 && cur->right == -1) {
      // is load pin, find from _node_map
      node = _node_map[cur->id];
      if(node==nullptr)std::cout<<"!!!\n";
      node->set_ptype(PinType::kLoad);
      // std::cout<<"cur->id "<<cur->id<<' '<<cur->p<<' '<<cur->left<<' '<<cur->right<<'\n';
    } else {
      // is steiner node
      node = new Node(std::to_string(cur->id), loc);
      _node_map.insert({cur->id, node});
    }
    parent_node->add_child(node);
    node->set_parent(parent_node);
    int direction;
    // std::cout<<"_areas "<<h_areas[cur->p].left<<'\n';
    if(h_areas[cur->p].left==cur->id)direction=0;
    else direction=1;
    // std::cout<<"_edge_len "<<parent->_edge_len[direction]<<'\n';
    auto edge_len = parent->_edge_len[direction];
    auto snake = edge_len - distanceB(parent->location, cur->location);
    node->set_required_snake(snake);
  }
}
}