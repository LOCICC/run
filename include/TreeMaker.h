#pragma once
#include "Instgpu.hh"
#include <stack>
#include "BoundSkewTree.h"
#include <cstdint>
#include <stdint.h>
#include <cstddef> 
#include "salt/salt_gpu.cuh"
#include <unordered_map>
#include <limits>
namespace gcts {  

class TreeMaker
{
    public:
    // TreeMaker(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound = std::nullopt,
    //   const std::optional<Point>& guide_loc = std::nullopt, const TopoType& topo_type = TopoType::kBiPartition);
    TreeMaker(Pointc* pts,Pointc* clusters,int* clusters_segment,std::vector<Inst*>& insts,int pinnum,int net_num,int t=3);
    TreeMaker(Pointc* pts,std::vector<Inst*>& insts,int pinnum,int t= 3);
    ~TreeMaker(){
        if(areas!=nullptr)cudaFree(areas);
        if(levelsum!=nullptr)cudaFree(levelsum);
    }
    // static Inst* cbsTree(const std::string& net_name, const std::vector<Pin*>& loads, const std::optional<double>& skew_bound,
    //                        const std::optional<Point>& guide_loc, const TopoType& topo_type = TopoType::kBiPartition);
    void cbsTree(Pointc* pts,Pointc* clusters,int* clusters_segment,std::vector<Inst*>& insts,int pinnum,int net_num);
    void cbsTreeBig(Pointc* pts,Pointc* clusters,int* clusters_segment,std::vector<Inst*>& insts,int pinnum,int net_num);
    static void SaltTreePool(std::vector<Inst*>& _root_buf,std::vector<Node*>& _root_buf_node,int i,int t);
    static void SaltTree(std::vector<Inst*>& _root_buf,std::vector<Node*>& _root_buf_node,int net_num,int t);
    static void SaltTreeGPU(std::vector<Inst*>& _root_buf,std::vector<Node*>& _root_buf_node,int net_num);
    static void saltgpu(int num_nets,std::vector<int>segments, std::vector<int>tree_start,std::vector<Point>pins,std::vector<Point>nodes,std::vector<int>_n,std::vector<int>roots,std::vector<bool>pin_ids);
    static void convertToBinaryTree(Node* root);
    static void removeRedundant(Node* root);
    static void disconnect(Node* parent, Node* child);
    static void connect(Node* parent, Node* child);
    static void Deletecpu(std::vector<Node*> _root_buf_node);
    static Inst* genBufInst(const std::string& prefix, const Point& location);
    static void updateId(Node* root);
    static void MinCostFlowRun(std::vector<Inst*>& insts,Pointc* pts,Pointc* clusters,Pointc*buffers,int max_fanout,int cluster_num);
    double* net_len=nullptr;
    double* net_delay=nullptr;
    double salttime;
    Area* areas=nullptr;
    int* levelsum=nullptr;
    int maxlevel;
    int type=3;
    
};
}