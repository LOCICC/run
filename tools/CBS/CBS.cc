
#include "CBS.h"
#include "ThreadPool.h"
#include "base/rsa.h"
#include "refine/refine.h"
#include <chrono>
namespace gcts {
void CBSInterface::init(salt::Tree& min_tree, std::shared_ptr<salt::Pin> src_pin)
{
  min_tree.UpdateId();
  auto mt_nodes = min_tree.ObtainNodes();
  _nodes.resize(mt_nodes.size());
  _shortest_latency.resize(mt_nodes.size());
  _cur_latency.resize(mt_nodes.size());
  for (auto mt_node : mt_nodes) {
    _nodes[mt_node->id] = std::make_shared<salt::TreeNode>(mt_node->loc, mt_node->pin, mt_node->id);
    _shortest_latency[mt_node->id] = utils::Dist(src_pin->loc, mt_node->loc);
    _cur_latency[mt_node->id] = std::numeric_limits<double>::max();
  }
  _cur_latency[src_pin->id] = 0;
  _src = _nodes[src_pin->id];
}

void CBSInterface::finalize(const salt::Net& net, salt::Tree& tree)
{
  for (auto n : _nodes) {
    if (n->parent) {
      _nodes[n->parent->id]->children.push_back(n);
    }
  }
  tree.source = _src;
  tree.net = &net;
}
int preOrder(const shared_ptr<salt::TreeNode> root, int up){
  int num=1;
  for(int i=0;i<root->children.size();i++) {
    num+=preOrder(root->children[i],up);
    if(num>=up)return up;
  }
  return num;
}
void mulsubstitutePool(salt::Tree & tree,vector<shared_ptr<salt::TreeNode>>subtree,int i,int pinnum,double eps){
  // std::cout<<subtree[i]->loc<<' '<<subtree[i]->id<<'\n';
  int pinids[pinnum];
        if(subtree[i]->children.size()==0)return;
        auto parent = subtree[i]->parent;
        subtree[i]->parent=nullptr;
        int num=0;
        auto T=tree;
        T.source=subtree[i];
        // tree.source=subtree[i];
        std::vector<std::shared_ptr<salt::Pin>> pins;
        int cc=0,tpin=0;
        T.preOrder([&](const shared_ptr<salt::TreeNode>& node) {
          if (node->pin) {
            // auto salt_pin = std::make_shared<salt::Pin>(node->pin->loc, num++, node->pin->cap);
            pinids[num++]=node->pin->id;
            pins.push_back(node->pin);
            node->pin->id=num-1;
            // node->pin=salt_pin;
            tpin++;
          }
          else if(node->parent==nullptr){
            auto salt_pin = std::make_shared<salt::Pin>(node->loc, num++, 0);
            pins.push_back(salt_pin);
            node->pin= salt_pin;
          }
          cc++;
        });
        salt::Net net1;
        net1.init(0, "tmp", pins);
        salt::Tree Subtree(subtree[i],&net1);
        // std::cout<<pins.size()<<' '<<num<<' '<<cc<<' '<<tpin<<'\n';
        Subtree.UpdateId();
        salt::Refine::substitute(Subtree, eps, true);
        subtree[i]->parent=parent;
        // std::cout<<subtree[i]->loc<<' '<<subtree[i]->children.size()<<' '<<subtree[i]->id<<'\n';
        if(subtree[i]->pin->cap==0&&i!=0)subtree[i]->pin=nullptr;
        Subtree.preOrder([&](const shared_ptr<salt::TreeNode>& node) {
          if (node->pin) node->pin->id=pinids[node->pin->id];
          //  std::cout<<node->id<<' '<<node->loc<<' '<<node->children.size()<<'\n';
        });
}
void mulsubstitute(salt::Tree & tree, double eps, int refine_level){
  vector<shared_ptr<salt::TreeNode>>subtree;
      subtree.push_back(tree.source);
      // std::cout<<tree.source->loc<<"\n";
      int c=0,c1=1;
      while(c1-c<50&&c<c1){
        for(int i=c;i<c1;i++){
          // std::cout<<subtree[i]->loc<<' '<<subtree[i]->id<<'\n';
          for(int j=0;j<subtree[i]->children.size();j++){
            if(preOrder(subtree[i]->children[j],100)>=100)subtree.push_back(subtree[i]->children[j]);
            // std::cout<<preOrder(subtree[i]->children[j],1000)<<'\n';
            // std::cout<<subtree[i]->children[j]->loc<<' '<<subtree[i]->children[j]->id<<'\n';
          }
        }
        c=c1;
        c1=subtree.size();
        // std::cout<<c<<' '<<c1<<'\n';
      }
      // int pinids[tree.net->pins.size()];
      // std::cout<<c<<' '<<c1<<'\n';
      ThreadPool pool(8);
      std::vector<std::future<bool>> futures;
      futures.reserve(c1-c);  // 预分配空间
      for(int i=c;i<c1;i++){
            futures.emplace_back(
                pool.enqueue(
                    [&subtree,&tree,i,eps] {
                        mulsubstitutePool(tree,subtree,i,tree.net->pins.size(),eps);
                        return true;
                    }
                )
            );
      }

        // 5. 等待所有任务完成（替代waitAll）
        for(auto& fut : futures) {
            fut.get();  // 阻塞直到任务完成
        }
        
      tree.source=subtree[0];
      tree.UpdateId();
      // tree.preOrder([&](const shared_ptr<salt::TreeNode>& node) {
      //     std::cout<<node->loc<<' '<<node->children.size()<<'\n';
      //   });
}
void TreeSaltBuilder::rungpu(const salt::Net& net, salt::Tree& input_tree, double eps, int refine_level)
{
  auto start = std::chrono::high_resolution_clock::now();
  // input tree
  auto tree = input_tree;
  // Refine tree
  if (refine_level >= 1) {
    salt::Refine::flip(tree);
    salt::Refine::uShift(tree);
    salt::Refine::removeRedundantCoincident(tree);
  }
  // std::cout<<refine_level<<'\n';
  // init
  init(tree, net.source());
  // dfs
  dfs(tree.source, _src, eps);
  finalize(net, input_tree);
  input_tree.RemoveTopoRedundantSteiner();

  // Connect breakpoints to source by RSA
  salt::RsaBuilder rsa_builder;
  rsa_builder.ReplaceRootChildren(input_tree);
  input_tree.UpdateId();
  // Refine SALT
  if (refine_level >= 1) {
    salt::Refine::cancelIntersect(input_tree);
    salt::Refine::flip(input_tree);
    salt::Refine::uShift(input_tree);
    auto start2 = std::chrono::high_resolution_clock::now();
    if (refine_level >= 2) {
      mulsubstitute(input_tree, eps, refine_level == 3);
      // input_tree.preOrder([&](const shared_ptr<salt::TreeNode>& node) {
      //   std::cout<<node->id<<' '<<node->loc<<'\n';
      // });
      // salt::Refine::substitute(input_tree, eps, refine_level == 3);
    }
  }
}
void TreeSaltBuilder::run(const salt::Net& net, salt::Tree& input_tree, double eps, int refine_level)
{
  // input tree
  auto tree = input_tree;
  // Refine tree
  if (refine_level >= 1) {
    salt::Refine::flip(tree);
    salt::Refine::uShift(tree);
    salt::Refine::removeRedundantCoincident(tree);
  }

  // init
  init(tree, net.source());
  // dfs
  dfs(tree.source, _src, eps);
  finalize(net, input_tree);
  input_tree.RemoveTopoRedundantSteiner();

  // Connect breakpoints to source by RSA
  salt::RsaBuilder rsa_builder;
  rsa_builder.ReplaceRootChildren(input_tree);
  input_tree.UpdateId();
  // Refine SALT
  if (refine_level >= 1) {
    salt::Refine::cancelIntersect(input_tree);
    salt::Refine::flip(input_tree);
    salt::Refine::uShift(input_tree);
    if (refine_level >= 2) {
      salt::Refine::substitute(input_tree, eps, refine_level == 3);
    }
  }
}
bool TreeSaltBuilder::relax(const std::shared_ptr<salt::TreeNode>& u, const std::shared_ptr<salt::TreeNode>& v)
{
  auto new_latency = _cur_latency[u->id] + utils::Dist(u->loc, v->loc);
  if (_cur_latency[v->id] > new_latency) {
    _cur_latency[v->id] = new_latency;
    v->parent = u;
    return true;
  }
  if (_cur_latency[v->id] == new_latency && utils::Dist(u->loc, v->loc) < v->WireToParentChecked()) {
    v->parent = u;
    return true;
  }
  return false;
}
void TreeSaltBuilder::dfs(const std::shared_ptr<salt::TreeNode>& tree_node, const std::shared_ptr<salt::TreeNode>& cbs_node, double eps)
{
  if (tree_node->pin && _cur_latency[cbs_node->id] > (1 + eps) * _shortest_latency[cbs_node->id]) {
    cbs_node->parent = _src;
    _cur_latency[cbs_node->id] = _shortest_latency[cbs_node->id];
  }
  for (auto c : tree_node->children) {
    relax(cbs_node, _nodes[c->id]);
    dfs(c, _nodes[c->id], eps);
    relax(_nodes[c->id], cbs_node);
  }
}

void TempBuilder::run(const salt::Net& net, salt::Tree& input_tree, double eps, int refine_level)
{
  // tree
  auto tree = input_tree;

  // Refine tree
  if (refine_level >= 1) {
    salt::Refine::flip(tree);
    salt::Refine::uShift(tree);
  }

  // init
  init(tree, net.source());

  // dfs
  dfs(tree.source, _src, eps);
  finalize(net, input_tree);
  input_tree.RemoveTopoRedundantSteiner();

  // Connect breakpoints to source by RSA
  salt::RsaBuilder rsa_builder;
  rsa_builder.ReplaceRootChildren(input_tree);

  // Refine SALT
  if (refine_level >= 1) {
    salt::Refine::cancelIntersect(input_tree);
    salt::Refine::flip(input_tree);
    salt::Refine::uShift(input_tree);
    if (refine_level >= 2) {
      salt::Refine::substitute(input_tree, eps, refine_level == 3);
    }
  }
}

void TempBuilder::init(salt::Tree& min_tree, std::shared_ptr<salt::Pin> src_pin)
{
  min_tree.UpdateId();
  auto mt_nodes = min_tree.ObtainNodes();
  _nodes.resize(mt_nodes.size());
  _shortest_latency.resize(mt_nodes.size());
  _cur_latency.resize(mt_nodes.size());
  // update cap load
  _cap_loads.resize(mt_nodes.size());
  min_tree.postOrder([&](const std::shared_ptr<salt::TreeNode>& n) {
    if (n->pin) {
      _cap_loads[n->id] = n->pin->cap;
      return;
    }
    _cap_loads[n->id] = 0;
    std::ranges::for_each(n->children, [&](const std::shared_ptr<salt::TreeNode>& c) { _cap_loads[n->id] += _cap_loads[c->id]; });
  });

  for (auto mt_node : mt_nodes) {
    _nodes[mt_node->id] = std::make_shared<salt::TreeNode>(mt_node->loc, mt_node->pin, mt_node->id);
    _shortest_latency[mt_node->id] = utils::Dist(src_pin->loc, mt_node->loc);
    _cur_latency[mt_node->id] = std::numeric_limits<double>::max();
  }
  _cur_latency[src_pin->id] = 0;
  _src = _nodes[src_pin->id];
}

void TempBuilder::resetParent(const std::shared_ptr<salt::TreeNode>& child, const std::shared_ptr<salt::TreeNode>& parent)
{
  auto origin_parent = child->parent;
  auto cap_load = _cap_loads[child->id];
  if (origin_parent) {
    _cap_loads[origin_parent->id] -= cap_load;
  }
  child->parent = parent;
  _cap_loads[parent->id] += cap_load;
}

bool TempBuilder::relax(const std::shared_ptr<salt::TreeNode>& u, const std::shared_ptr<salt::TreeNode>& v)
{
  auto new_latency = _cur_latency[u->id] + delay(u, v);
  if (_cur_latency[v->id] > new_latency) {
    _cur_latency[v->id] = new_latency;
    resetParent(v, u);
    return true;
  }
  auto delay_from_parent = v->parent ? delay(v->parent, v) : 0;
  if (_cur_latency[v->id] == new_latency && delay(u, v) < delay_from_parent) {
    resetParent(v, u);
    return true;
  }
  return false;
}

void TempBuilder::dfs(const std::shared_ptr<salt::TreeNode>& tree_node, const std::shared_ptr<salt::TreeNode>& cbs_node, double eps)
{
  if (tree_node->pin && _cur_latency[cbs_node->id] > (1 + eps) * _shortest_latency[cbs_node->id]) {
    resetParent(cbs_node, _src);
    _cur_latency[cbs_node->id] = _shortest_latency[cbs_node->id];
  }
  for (auto c : tree_node->children) {
    relax(cbs_node, _nodes[c->id]);
    dfs(c, _nodes[c->id], eps);
    relax(_nodes[c->id], cbs_node);
  }
}
}  // namespace icts