#include "salt.h"
#include "TreeMaker.h"
#include "CBS.h"
#include "ThreadPool.h"
#include "Timing.h"
using namespace gcts;

TreeMaker::TreeMaker(Pointc* pts,Pointc* clusters,int* clusters_segment,std::vector<Inst*>& insts,int pinnum,int net_num,int t){
  cudaMalloc(&net_len,net_num*sizeof(double));
  cudaMalloc(&net_delay,net_num*sizeof(double));
  // std::cout<<"cbsTree begin\n";
  type=t;
  cbsTree(pts,clusters,clusters_segment,insts,pinnum,net_num);
}
TreeMaker::TreeMaker(Pointc* pts,std::vector<Inst*>& insts,int pinnum,int t){
  cudaMalloc(&net_len,sizeof(double));
  cudaMalloc(&net_delay,sizeof(double));
  Pointc h_clusters(0,0,pinnum,0);
  Pointc* clusters;
  int *clusters_segment;
  cudaMalloc(&clusters,sizeof(Pointc));
  cudaMalloc(&clusters_segment,sizeof(int));
  cudaMemcpy(clusters, &h_clusters, sizeof(Pointc), cudaMemcpyHostToDevice);
  cudaMemset(clusters_segment, 0, sizeof(int));
  type=t;
  cbsTree(pts,clusters,clusters_segment,insts,pinnum,1);
}
void TreeMaker::cbsTreeBig(Pointc* pts,Pointc* clusters,int* clusters_segment,std::vector<Inst*>& insts,int pinnum,int net_num)
{
//   if (loads.size() == 1) {
//     auto loc = guide_loc.value_or(loads.front()->get_location());
//     auto* buf = genBufInst(net_name, loc);
//     directConnectTree(buf->get_driver_pin(), loads.front());
//     return buf;
//   }
  // std::cout<<"cbsTree\n";
    // build BST
  cudaDeviceSynchronize();
  auto startbst = std::chrono::high_resolution_clock::now();
  auto bst = BoundSkewTree(pts,clusters,clusters_segment,pinnum,net_num);
  bst.big=true;
  bst.run();
  bst.recursiveBottomUp();
  bst.embedding();
  bst.convert();
  // bst.calcNetLen();
  // bst.estimateNetDelay();
  cudaDeviceSynchronize();
  auto endbst = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedbst = endbst - startbst;
  // std::cout << "BoundSkewTree time: " << elapsedbst.count() << " seconds" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  bst.GPUtoCPU(pts);
  cudaDeviceSynchronize();
  for(int i=0;i<net_num;i++)removeRedundant(bst._root_buf_node[i]);//可不要
  Timing::update(bst._root_buf_node,insts,bst.load_nodes,net_num); 
  SaltTree(bst._root_buf,bst._root_buf_node,net_num,type);
  // SaltTreeGPU(bst._root_buf,bst._root_buf_node,net_num);
  auto ends = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapseds = ends - start;
  // std::cout << "SaltTree time: " << elapseds.count() << " seconds" << std::endl;

  for(int i=0;i<net_num;i++)convertToBinaryTree(bst._root_buf_node[i]);
  for(int i=0;i<net_num;i++)bst.runTwo(i,true,0.08);
  bst.CPUtoGPU(pts);
  // cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  salttime=elapsed.count();
  // std::cout << "ALLSaltTree time: " << elapsed.count() << " seconds" << std::endl;
  // auto solver = BoundSkewTree(pts,clusters,clusters_segment,pinnum,net_num);
  bst.recursiveBottomUp();
  bst.embedding();
  bst.convert();
  bst.calcNetLen();
  // bst.estimateNetDelay();
  // Deletecpu(bst._root_buf_node);
  cudaDeviceSynchronize();
  if(bst.net_len!=nullptr)cudaMemcpy(net_len,bst.net_len,net_num*sizeof(double),cudaMemcpyDeviceToDevice);
  if(bst.net_delay!=nullptr)cudaMemcpy(net_delay,bst.net_delay,net_num*sizeof(double),cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  areas = bst.areas;
  levelsum = bst.levelsum;
  maxlevel = bst.maxlevel;
  // coutDouble(net_len,net_num);
  // std::cout<<"over\n";
}
void TreeMaker::Deletecpu(std::vector<Node*> _root_buf_node){
  for(int i=0;i< _root_buf_node.size();i++){
    _root_buf_node[i]->postOrder([&](Node* node){delete node; });
  }
}
void coutDouble(double* d,int size){
  double *h = new double[size];
  cudaMemcpy(h, d, (size) * sizeof(double), cudaMemcpyDeviceToHost);
  std::cout << "values: ";
  for (int i = 0; i < size ; i++) {
      std::cout << h[i] << " ";
  }
  std::cout << std::endl;
}
void TreeMaker::cbsTree(Pointc* pts,Pointc* clusters,int* clusters_segment,std::vector<Inst*>& insts,int pinnum,int net_num)
{
  cudaDeviceSynchronize();
  auto startbst = std::chrono::high_resolution_clock::now();
  auto bst = BoundSkewTree(pts,clusters,clusters_segment,pinnum,net_num);
  
  if(net_num==1)bst.big=true;
  bst.run();
  
  bst.recursiveBottomUp();
  // std::cout<<"recursiveBottomUp\n";
  bst.embedding();
  // std::cout<<"embedding\n";
  bst.convert();
  // std::cout<<"convert\n";
  bst.calcNetLen();
  cudaDeviceSynchronize();
  // bst.estimateNetDelay();
  // cudaDeviceSynchronize();
  auto endbst = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedbst = endbst - startbst;
  std::cout << "BoundSkewTree time: " << elapsedbst.count() << " seconds" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  bst.GPUtoCPU(pts);//
  // std::cout << "here" << std::endl;//
  for(int i=0;i<net_num;i++)removeRedundant(bst._root_buf_node[i]);//可不要//
  // std::cout<<"update\n";
  Timing::update(bst._root_buf_node,insts,bst.load_nodes,net_num);  
  SaltTree(bst._root_buf,bst._root_buf_node,net_num,type);//
  // SaltTreeGPU(bst._root_buf,bst._root_buf_node,net_num);
  cudaDeviceSynchronize();
  auto ends = std::chrono::high_resolution_clock::now();//
  std::chrono::duration<double> elapseds = ends - start;//
  std::cout << "SaltTree time: " << elapseds.count() << " seconds" << std::endl;

  for(int i=0;i<net_num;i++)convertToBinaryTree(bst._root_buf_node[i]);
  for(int i=0;i<net_num;i++)bst.runTwo(i,true,0.08);
  bst.CPUtoGPU(pts);
  // cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  salttime=elapsed.count();
  // std::cout << "ALLSaltTree time: " << elapsed.count() << " seconds" << std::endl;
  // auto solver = BoundSkewTree(pts,clusters,clusters_segment,pinnum,net_num);
  bst.recursiveBottomUp();
  bst.embedding();
  bst.convert();
  bst.calcNetLen();  
  // bst.estimateNetDelay();
  cudaDeviceSynchronize();
  if(bst.net_len!=nullptr)cudaMemcpy(net_len,bst.net_len,net_num*sizeof(double),cudaMemcpyDeviceToDevice);
  if(bst.net_delay!=nullptr)cudaMemcpy(net_delay,bst.net_delay,net_num*sizeof(double),cudaMemcpyDeviceToDevice);
  areas = bst.areas;
  levelsum = bst.levelsum;
  maxlevel = bst.maxlevel;
  // coutDouble(net_len,net_num);
  // std::cout<<"over\n";
}
void TreeMaker::disconnect(Node* parent, Node* child)
{
  parent->remove_child(child);
  child->set_parent(nullptr);
}
void TreeMaker::connect(Node* parent, Node* child)
{
  parent->add_child(child);
  child->set_parent(parent);
}
void TreeMaker::removeRedundant(Node* root){
  std::vector<Node*> to_be_removed;
  std::stack<Node*> stack;
  stack.push(root);
  while (!stack.empty()) {
    auto* node = stack.top();
    stack.pop();
    auto children = node->get_children();
    bool exist_opt = false;
    for (auto* child : children) {
      if (node->get_location() == child->get_location()) {
        exist_opt = true;
        break;
      }
    }
    if (!exist_opt) {
      std::ranges::for_each(children, [&](Node* child) { stack.push(child); });
      continue;
    }
    // move target child's children to node, and remove target child
    int cnt=0;
    if (node->isPin()) {
      cnt++;
      std::ranges::for_each(children, [&](Node* child) {
        if (node->get_location() != child->get_location()) {
          return;
        }
        // move child's children to node, and add child to to_be_removed
        disconnect(node, child);
        auto sub_children = child->get_children();
        std::ranges::for_each(sub_children, [&](Node* sub_child) {
          disconnect(child, sub_child);
          connect(node, sub_child);
        });
        to_be_removed.push_back(child);
      });
      stack.push(node);  // recheck new children whether have same location
      continue;
    }
    // let target child be new node
    for (auto* child : children) {
      if (node->get_location() != child->get_location()) {
        continue;
      }
      auto* parent = node->get_parent();
      disconnect(node, child);
      disconnect(parent, node);
      connect(parent, child);
      std::ranges::for_each(children, [&](Node* sub_child) {
        if (sub_child == child) {
          return;
        }
        disconnect(node, sub_child);
        connect(child, sub_child);
      });
      stack.push(child);
      node->set_children({});
      to_be_removed.push_back(node);
      break;
    }
  }
  std::ranges::for_each(to_be_removed, [&](Node* node) { delete node; });
}
void TreeMaker::updateId(Node* root){
  std::vector<Node*> pin_nodes;
  std::vector<Node*> steiner_nodes;
  root->preOrder([&](Node* node) {
    if (node->isPin()) {
      pin_nodes.push_back(node);
    } else {
      steiner_nodes.push_back(node);
    }
  });
  int id = 0;
  std::ranges::for_each(pin_nodes, [&](Node* node) { node->set_id(id++); });
  std::ranges::for_each(steiner_nodes, [&](Node* node) {
    node->set_id(id++);
    node->set_name("steiner_" + std::to_string(node->get_id()));
  });
}
Inst* TreeMaker::genBufInst(const std::string& prefix, const Point& location){
  auto buf_name = prefix + "_buf";
  auto buf_inst = new Inst(buf_name, location, InstType::kBuffer);
  return buf_inst;
}
void TreeMaker::convertToBinaryTree(Node* root){
  auto id = root->getMaxId();
  auto pin_node_refine = [&](Node* node) {
    if (!node->isPin() || !node->isLoad()) {
      return node;
    }
    auto children = node->get_children();
    if (children.empty()) {
      return node;
    }
    auto* parent = node->get_parent();
    auto* copy_node = new Node(++id, node->get_location());
    disconnect(parent, node);
    connect(parent, copy_node);
    connect(copy_node, node);

    // downstream refine
    std::ranges::for_each(children, [&](Node* child) {
      disconnect(node, child);
      connect(copy_node, child);
    });
    return copy_node;
  };

  // convert to binary tree for bound skew tree
  auto one_child_refine = [&](Node* node) {
    auto children = node->get_children();
    // case 1: size is 1
    auto* child = children.front();
    if ((node->isPin() && node->isLoad())) {
      // upstream refine
      auto* parent = node->get_parent();
      auto* copy_node = new Node(++id, node->get_location());
      if (parent) {
        disconnect(parent, node);
        connect(parent, copy_node);
      }
      // downstream refine
      disconnect(node, child);
      connect(copy_node, child);
      connect(copy_node, node);
      return copy_node;
    }
    auto grand_children = child->get_children();
    std::ranges::for_each(grand_children, [&](Node* grand_child) {
      disconnect(child, grand_child);
      connect(node, grand_child);
    });
    if (!child->isPin()) {
      disconnect(node, child);
      delete child;
    }
    return node;
  };

  auto three_child_refine = [&](Node* node) {
    auto children = node->get_children();
    // LOG_FATAL_IF(children.size() != 3) << "node " << node->get_name() << " children size is not 3";

    // case 3: size is 3

    // find closest 2 node in children TBD use heuristic
    std::ranges::sort(children, [&](const Node* lhs, const Node* rhs) {
      return Point::manhattanDistance(lhs->get_location(), node->get_location())
             < Point::manhattanDistance(rhs->get_location(), node->get_location());
    });
    // downstream refine
    auto* left_child = children[0];
    auto* right_child = children[1];
    auto* trunk = new Node(++id, node->get_location());
    disconnect(node, left_child);
    disconnect(node, right_child);
    connect(node, trunk);
    connect(trunk, left_child);
    connect(trunk, right_child);
    return node;
  };

  auto four_child_refine = [&](Node* node) {
    auto children = node->get_children();

    // upstream check
    // case 4: size is 4
    auto* copy_left_node = new Node(++id, node->get_location());
    auto* copy_right_node = new Node(++id, node->get_location());
    std::ranges::for_each(children, [&](Node* child) { disconnect(node, child); });
    connect(node, copy_left_node);
    connect(node, copy_right_node);
    // find closest 2 node in children TBD use heuristic
    std::ranges::sort(children, [&](const Node* lhs, const Node* rhs) {
      return Point::manhattanDistance(lhs->get_location(), node->get_location())
             < Point::manhattanDistance(rhs->get_location(), node->get_location());
    });
    // downstream refine
    auto left_children = std::vector<Node*>(children.begin(), children.begin() + 2);
    std::ranges::for_each(left_children, [&](Node* child) { connect(copy_left_node, child); });
    auto right_children = std::vector<Node*>(children.begin() + 2, children.end());
    std::ranges::for_each(right_children, [&](Node* child) { connect(copy_right_node, child); });
    return node;
  };

  auto to_binary = [&](Node* node) {
    node = pin_node_refine(node);
    auto children = node->get_children();
    if (children.empty()) {
      return node;
    }
    if (children.size() == 1) {
      return one_child_refine(node);
    }
    if (children.size() == 2) {
      return node;
    }
    if (children.size() == 3) {
      return three_child_refine(node);
    }
    return four_child_refine(node);
  };
  std::stack<Node*> stack;
  stack.push(root);
  while (!stack.empty()) {
    auto* cur = stack.top();
    stack.pop();
    // std::cout<<"ychildren:"<<cur->get_children().size()<<'\n';
    cur = to_binary(cur);
    if (cur == root) {
      // driver_pin should be checked
      cur = to_binary(cur);
    }
    auto children = cur->get_children();
    // std::cout<<"children:"<<children.size()<<'\n';
    if (children.empty()) {
      continue;
    }
    stack.push(children.front());
    stack.push(children.back());
  }
}                     
  // std::cout<<"BoundSkewTree\n";
void TreeMaker::SaltTreePool(std::vector<Inst*>& _root_buf,std::vector<Node*>& _root_buf_node,int i,int t){
  auto* buf = _root_buf[i];
  auto* driver_pin = buf->get_driver_pin();
  auto* driver_node = _root_buf_node[i];
  int num1=0;
  driver_node->preOrder([&](Node* node) { ++num1; });
  // std::cout<<num1<<'\n';
  // std::cout<<"salt\n";                         
  updateId(driver_node);
  int num = 0;
  driver_node->preOrder([&](Node* node) { ++num; });

  // std::cout<<"SaltTreePool:"<<num<<'\n';
  std::vector<std::shared_ptr<salt::Pin>> salt_pins;
  std::vector<Node*> cts_nodes(num);
  std::vector<std::shared_ptr<salt::TreeNode>> salt_nodes(num);
  // driver_node->preOrder([&](Node* node) { std::cout<<node->get_location().x()<<' '<<node->get_location().y()<<' '<<node->get_id()<<' '<<node->get_cap_load()<<'\n'; });  
  driver_node->preOrder([&](Node* node) {
    std::shared_ptr<salt::TreeNode> salt_node;
    auto id = node->get_id();
    // if(node->get_location().x()==799285&&node->get_location().y()==295060)std::cout<<node->get_location().x()<<' '<<node->get_location().y()<<'\n';
    auto loc = salt::Point(node->get_location().x(), node->get_location().y());
    if (node->isPin()) {
      auto salt_pin = std::make_shared<salt::Pin>(loc, id, node->_cap_load);
      salt_node = std::make_shared<salt::TreeNode>(loc, salt_pin, id);
      salt_pins.push_back(salt_pin);
    } else {
      salt_node = std::make_shared<salt::TreeNode>(loc, nullptr, id);
    }
    salt_nodes[id] = salt_node;
    cts_nodes[id] = node;
  });
  // std::cout<<salt_pins.size()<<'\n';
  salt::Net salt_net;
  salt_net.init(0, std::to_string(i), salt_pins);
  // convert bound skew tree to salt data structure
  driver_node->preOrder([&](Node* node) { 
    if (!node->get_parent()) {
      return;
    }
    auto cur_id = node->get_id();
    auto parent_id = node->get_parent()->get_id();
    auto salt_node = salt_nodes[cur_id];
    auto salt_parent = salt_nodes[parent_id];
    salt::TreeNode::setParent(salt_node, salt_parent);
  });
  // BST Salt
  salt::Tree bound_skew_tree(salt_nodes[0], &salt_net);
  TreeSaltBuilder builder;
  builder.rungpu(salt_net, bound_skew_tree, 0,t);
  // connect driver node to all loads based on salt's tree(node), if node not exist, create new node
  // TimingPropagator::resetNet(bst_net);
  Timing::resetNet(driver_node);
  auto source = bound_skew_tree.source;
  buf = genBufInst(std::to_string(i), Point(source->loc.x, source->loc.y));
  // driver_pin = buf->get_driver_pin();
  driver_node = new Node(buf->get_driver_pin()->get_name(),buf->get_driver_pin()->get_location());
  driver_node->set_type(NodeType::kBufferPin);
  driver_node->set_ptype(PinType::kDriver);
  cts_nodes[source->id] = driver_node;
  num = 0;
  auto count_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) { ++num; };
  salt::TreeNode::preOrder(source, count_func);
  cts_nodes.resize(num);
  // std::cout<<num<<'\n';
  auto connect_node_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) {
    // steiner point, need to create a new node
    auto id = salt_node->id;
    if (id == source->id) {
      return;
    }
    if (!salt_node->pin) {
      auto* node = new Node(id, Point(salt_node->loc.x, salt_node->loc.y));
      cts_nodes[id] = node;
    }
    // connect to parent
    auto* current_node = cts_nodes[id];
    if((!(!salt_node->pin))&&current_node->_ptype!=PinType::kDriver)current_node->set_ptype(PinType::kLoad);
    auto parent_id = salt_node->parent->id;
    auto* parent_node = cts_nodes[parent_id];
    connect(parent_node, current_node);
  };
  salt::TreeNode::preOrder(source, connect_node_func);
  _root_buf[i]=buf;
  _root_buf_node[i]=driver_node;
  // std::cout<<driver_node->get_location().x()<<' '<<driver_node->get_location().y()<<' '<<driver_node->get_cap_load()<<'\n';
  // _root_buf_node[i]->preOrder([&](Node* node) { std::cout<<node->get_location().x()<<' '<<node->get_location().y()<<' '<<node->get_cap_load()<<'\n'; });
  // }
}
void TreeMaker::SaltTreeGPU(std::vector<Inst*>& _root_buf,std::vector<Node*>& _root_buf_node,int net_num){
                
  std::vector<std::shared_ptr<salt::Pin>> salt_pins; 
  std::vector<Point>pins;
  std::vector<Point>nodes;
  std::vector<int>n;
  std::vector<int>roots;
  std::vector<int>segments;
  std::vector<int>tree_start;
  std::vector<bool>pin_ids;
  segments.push_back(0);
  tree_start.push_back(0);
int num=0;
for(int i=0;i<_root_buf.size();i++){
  auto* buf = _root_buf[i];
  auto* driver_pin = buf->get_driver_pin();
  auto* driver_node = _root_buf_node[i];
  updateId(driver_node);
  roots.push_back(driver_node->get_id()+num);
  int num1=0;
  driver_node->preOrder([&](Node* node) { ++num1; });
  // std::cout<<"root:"<<driver_node->get_location().x()<<' '<<driver_node->get_location().y()<<'\n';
  // std::cout<<num1<<'\n';
  std::vector<Point>nodes_tmp(num1);
  std::vector<int>n_tmp(num1);
  std::vector<bool>pin_id(num1);
  driver_node->preOrder([&](Node* node) {
    auto id = node->get_id();
    pin_id[id]=false;
    // nodes.push_back(Point(node->get_location().x(),node->get_location().y()));
    // if(node->get_parent()==nullptr)n.push_back(num+id);
    // else n.push_back(num+node->get_parent()->get_id());
    // if (node->isPin()&&node->get_parent()!=nullptr) {
    //   pins.push_back(Point(node->get_location().x(),node->get_location().y()));
    // }
    nodes_tmp[id]=Point(node->get_location().x(),node->get_location().y());
    if(node->get_parent()==nullptr)n_tmp[id]=id+num;
    else n_tmp[id]=num+node->get_parent()->get_id();
    if (node->isPin()) {
      pins.push_back(Point(node->get_location().x(),node->get_location().y()));
      if(node->get_parent()!=nullptr)pin_id[id]=true;
    }
  });
  num=nodes.size();
  pin_ids.insert(pin_ids.end(), pin_id.begin(), pin_id.end());
  nodes.insert(nodes.end(), nodes_tmp.begin(), nodes_tmp.end());
  n.insert(n.end(), n_tmp.begin(), n_tmp.end());
  segments.push_back(pins.size());
  tree_start.push_back(nodes.size());
}
  // std::cout<<salt_pins.size()<<'\n';
  // TreeMaker::saltgpu(net_num,segments,tree_start,pins,nodes,n,roots,pin_ids);
  // connect driver node to all loads based on salt's tree(node), if node not exist, create new node
}
void TreeMaker::SaltTree(std::vector<Inst*>& _root_buf,std::vector<Node*>& _root_buf_node,int net_num,int t){
  if(net_num==1){
    SaltTreePool(_root_buf, _root_buf_node,0,t);
    return;
  }
  ThreadPool pool(8);
    
    // 2. 存储future对象用于等待完成
    std::vector<std::future<bool>> futures;
    futures.reserve(net_num);  // 预分配空间

    // 3. 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 4. 提交任务
    for(int i = 0; i < net_num; i++) {
        futures.emplace_back(
            pool.enqueue(
                [&_root_buf,&_root_buf_node,i,t] {
                    SaltTreePool(_root_buf, _root_buf_node,i,t);
                    return true;
                }
            )
        );
    }

    // 5. 等待所有任务完成（替代waitAll）
    for(auto& fut : futures) {
        fut.get();  // 阻塞直到任务完成
    }

    // 6. 计算耗时
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "Total execution time: " << duration.count() << " ms" << std::endl;
}  

