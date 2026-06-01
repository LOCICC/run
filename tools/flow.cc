#include "CTS.h"
#include "BoundSkewTree.h"
#include "GPUClustering.h"
#include "Timing.h"
#include "CtsCellLib.hh"
#include "LevelSolve.h"
#include "TreeMaker.h"
using namespace gcts;
int PreOrder(const Area *hareas,int &cnt,int pre, int id, int* &left_child, int* &right_child, int* &tree_node_x, int* &tree_node_y) {
    int I=cnt;
    // std::cout<<hareas[id].location.x<<' '<<hareas[id].location.y<<'\n';
    tree_node_x[I]=hareas[id].location.x;
    tree_node_y[I]=hareas[id].location.y;
    cnt++;
    if(hareas[id].left!=-1)left_child[I]=PreOrder(hareas,cnt,pre,hareas[id].left,left_child,right_child,tree_node_x,tree_node_y);
    else left_child[I]=-1;
    if(hareas[id].right!=-1)right_child[I]=PreOrder(hareas,cnt,pre,hareas[id].right,left_child,right_child,tree_node_x,tree_node_y);
    else right_child[I]=-1;
    return I-pre;
}
void BoundSkewTreeRun(const int net_num, const int* load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const double skew_bound, 
    int*& root, int*& left_child, int*& right_child, int*& tree_node_x, int*& tree_node_y,int* &tree_node_num)
{
        if(Timing::_db_unit==0)Timing::init();
        int* clusters_segment;
        cudaMalloc(&clusters_segment,net_num*sizeof(int));
        int clusters_segment_1[net_num];
        Pointc clusters1[net_num];
        for(int i=0;i<net_num;i++){
            clusters1[i].count=load_pins_num[i];
            if(i==0)clusters_segment_1[i]=0;
            else clusters_segment_1[i]=clusters_segment_1[i-1]+load_pins_num[i-1];
        }
        int pin_num=clusters_segment_1[net_num-1]+load_pins_num[net_num-1];
        cudaMemcpy(clusters_segment,clusters_segment_1,net_num*sizeof(int),cudaMemcpyHostToDevice);
        Pointc *cpts=new Pointc[pin_num];
        Pointc *pts;
        int cnt=0;
        std::cout<<"ree\n";
        for(int i=0;i<pin_num;i++){
            cpts[i]=Pointc(load_pins_x[i],load_pins_y[i]);
            cpts[i].cap=load_pins_cap[i];
            // std::cout<<i<<'\n';
        }
        std::cout<<"ree\n";
        cudaMalloc(&pts,pin_num*sizeof(Pointc));
        cudaMemcpy(pts, cpts, pin_num*sizeof(Pointc), cudaMemcpyHostToDevice);
        Pointc* clusters;
        cudaMalloc(&clusters,net_num*sizeof(Pointc));
        cudaMemcpy(clusters,clusters1,net_num*sizeof(Pointc),cudaMemcpyHostToDevice);
        if(left_child==nullptr)left_child=new int[pin_num*3];
        if(right_child==nullptr)right_child=new int[pin_num*3];
        if(tree_node_x==nullptr)tree_node_x=new int[pin_num*3];
        if(tree_node_y==nullptr)tree_node_y=new int[pin_num*3];
        if(root==nullptr)root=new int[pin_num*3];
        if(tree_node_num==nullptr)tree_node_num=new int[net_num+1];  
        auto bst = BoundSkewTree(pts,clusters,clusters_segment,pin_num,net_num);
        if(net_num==1)bst.big=true;
        bst.build(80.,1000,0.,0.,0.,0.);
        Area *hareas=new Area[3*pin_num];
        cudaMemcpy(hareas,bst.areas,3*pin_num*sizeof(Area),cudaMemcpyDeviceToHost);
        cudaFree(bst.areas);
        int c=0;
        for(int i=0;i<net_num;i++){
            int pre=c;
            root[i]=PreOrder(hareas,c,pre,pin_num+i,left_child,right_child,tree_node_x,tree_node_y);
            // std::cout<<c<<'\n';
            tree_node_num[i+1]=c;
        }
}
void BoundSkewTreeSingleRun(const int load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const double skew_bound, 
    int& root, int*& left_child, int*& right_child, int*& tree_node_x, int*& tree_node_y,int &tree_node_num)
{
        if(Timing::_db_unit==0)Timing::init();
        Pointc *cpts=new Pointc[load_pins_num];
        Pointc *pts;
        int cnt=0;
        for(int i=0;i<load_pins_num;i++){
            cpts[i]=Pointc(load_pins_x[i]/Timing::_db_unit,load_pins_y[i]/Timing::_db_unit);
            cpts[i].cap=load_pins_cap[i];
        }
        cudaMalloc(&pts,load_pins_num*sizeof(Pointc));
        cudaMemcpy(pts, cpts, load_pins_num*sizeof(Pointc), cudaMemcpyHostToDevice);
        Pointc* clusters;
        cudaMalloc(&clusters,sizeof(Pointc));
        Pointc clusters1;
        clusters1.count=load_pins_num;
        cudaMemcpy(clusters,&clusters1,sizeof(Pointc),cudaMemcpyHostToDevice);
        int *clusters_segment;
        cudaMalloc(&clusters_segment,2*sizeof(int));
        int clusters_segment_1[2];
        clusters_segment_1[0]=0,clusters_segment_1[1]=load_pins_num;
        cudaMemcpy(clusters_segment,clusters_segment_1,2*sizeof(int),cudaMemcpyHostToDevice);
        if(left_child==nullptr)left_child=new int[load_pins_num*3];
        if(right_child==nullptr)right_child=new int[load_pins_num*3];
        if(tree_node_x==nullptr)tree_node_x=new int[load_pins_num*3];
        if(tree_node_y==nullptr)tree_node_y=new int[load_pins_num*3];
        int net_num=1;
        auto bst = BoundSkewTree(pts,clusters,clusters_segment,load_pins_num,net_num);
        if(net_num==1)bst.big=true;
        bst.build(skew_bound,Timing::_db_unit,0.,0.,0.,0.);
        bst.calcNetLen();
        Area *hareas=new Area[3*load_pins_num];
        cudaMemcpy(hareas,bst.areas,3*load_pins_num*sizeof(Area),cudaMemcpyDeviceToHost);
        cudaFree(bst.areas);
        int c=0;
        root=PreOrder(hareas,c,0,load_pins_num,left_child,right_child,tree_node_x,tree_node_y);
        tree_node_num=c;
}
void IterClusteringRun(const int load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const size_t& max_fanout,
                           const size_t& iters, const size_t& no_change_stop, int &cluster_num, int *&clusters_segment, int *&load_pins_index)
{
    Pointc* pts;
    Pointc* clusters;
    if(Timing::_db_unit==0)Timing::init();
    std::vector<Inst*>insts;
    int c=0;
    for(int i=0;i<load_pins_num;i++){
        Point location(load_pins_x[i]*1./Timing::_db_unit, load_pins_y[i]*1./Timing::_db_unit);  // 假设 Point 有 (x,y) 构造函数
        Inst* inst = new Inst(std::to_string(c++), location);
        inst->cap=load_pins_cap[i];
        inst->ID=i;
        insts.push_back(inst);
    }
    Pointc *cpts=new Pointc[load_pins_num];
    int cnt=0;
    for(int i=0;i<load_pins_num;i++){
        cpts[i]=Pointc(insts[i]->_location.x(),insts[i]->_location.y());
        cpts[i].cap=insts[i]->cap;
        cpts[i].instID=i;
    }
    cudaMalloc(&pts,load_pins_num*sizeof(Pointc));
    cudaMemcpy(pts, cpts, load_pins_num*sizeof(Pointc), cudaMemcpyHostToDevice);
    cudaMalloc(&clusters,load_pins_num*sizeof(Pointc));
    auto clu=GPUClustering();
    clu.INIT(load_pins_num);
    std::cout<<"------ start IterClustering ------\n";
    cluster_num = clu.iterClustering(insts,pts,clusters,max_fanout,iters,5);
    Pointc* h_clusters=new Pointc[cluster_num];
    cudaMemcpy(cpts,pts,load_pins_num*sizeof(Pointc),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clusters,clusters,cluster_num*sizeof(Pointc),cudaMemcpyDeviceToHost);
    if(clusters_segment==nullptr)clusters_segment = new int[cluster_num+1];
    if(load_pins_index==nullptr)load_pins_index = new int[load_pins_num];
    clusters_segment[0]=0;
    for(int i=0;i<cluster_num;i++)clusters_segment[i+1]=clusters_segment[i]+h_clusters[i].count;
    for(int i=0;i<load_pins_num;i++)load_pins_index[i]=cpts[i].instID;
    std::cout<<"------ end IterClustering ------\n";
    // for(int i=0;i<20;i++)std::cout<<cpts[i].instID<<' ';
}

void EnhanceClusteringRun(const int load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const size_t max_fanout,
                          const int cluster_num, const int *clusters_segment, const int max_len, const size_t& iters,
                        int &next_cluster_num, int *&next_clusters_segment, int *&next_load_pins_index)
{
    Pointc* pts;
    Pointc* clusters;
    int *clusters_segment_gpu,clusters_segment_cpu[cluster_num];
    if(Timing::_db_unit==0)Timing::init();
    std::vector<Inst*>insts;
    int c=0;
    for(int i=0;i<load_pins_num;i++){
        Point location(load_pins_x[i], load_pins_y[i]);  // 假设 Point 有 (x,y) 构造函数
        Inst* inst = new Inst(std::to_string(c++), location);
        inst->cap=load_pins_cap[i];
        insts.push_back(inst);
    }
    Pointc *cpts=new Pointc[load_pins_num];
    int cnt=0;
    for(int i=0;i<load_pins_num;i++){
        cpts[i]=Pointc(insts[i]->_location.x(),insts[i]->_location.y());
        cpts[i].cap=insts[i]->cap;
        cpts[i].instID=i;
    }
    Pointc *cclusters=new Pointc[cluster_num];
    for(int i=0;i<cluster_num;i++){
        cclusters[i].count=clusters_segment[i];
        if(i==0)clusters_segment_cpu[i]=0;
        else clusters_segment_cpu[i]=clusters_segment[i-1]+clusters_segment_cpu[i-1];
    }
    cudaMalloc(&pts,load_pins_num*sizeof(Pointc));
    cudaMemcpy(pts, cpts, load_pins_num*sizeof(Pointc), cudaMemcpyHostToDevice);
    cudaMalloc(&clusters,load_pins_num*sizeof(Pointc));
    cudaMalloc(&clusters_segment_gpu,cluster_num*sizeof(int));

    cudaMemcpy(clusters,cclusters,cluster_num*sizeof(Pointc),cudaMemcpyHostToDevice);
    cudaMemcpy(clusters_segment_gpu,clusters_segment_cpu,cluster_num*sizeof(int),cudaMemcpyHostToDevice);
    auto clu=GPUClustering();
    clu.INIT(load_pins_num);
    int num = clu.slackClustering(pts,clusters,clusters_segment_gpu,insts,max_len,cluster_num);
    next_cluster_num = num;
    Pointc* h_clusters=new Pointc[cluster_num];
    cudaMemcpy(cpts,pts,load_pins_num*sizeof(Pointc),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clusters,clusters,num*sizeof(Pointc),cudaMemcpyDeviceToHost);
    if(next_clusters_segment==nullptr)next_clusters_segment = new int[num+1];
    if(next_load_pins_index==nullptr)next_load_pins_index = new int[load_pins_num];
    next_clusters_segment[0]=0;
    for(int i=0;i<num;i++)next_clusters_segment[i+1]=next_clusters_segment[i]+h_clusters[i].count;
    for(int i=0;i<load_pins_num;i++)next_load_pins_index[i]=cpts[i].instID;
}

void LevelRun(const int load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const int max_fanout, const double max_len,
    const int fixed_num, const int* fixed_x,const int* fixed_y,
    int &next_cluster_num, int *&next_clusters_segment, int *&next_load_pins_index,
    int &next_level_load_pins_num, int *&next_level_load_pins_x, int *&next_level_load_pins_y, double *&next_level_load_pins_cap, std::string *&buffer_cell_masters)
{   if(Timing::_db_unit==0)Timing::init();
    std::vector<Inst*>insts;
    int c=0;
    for(int i=0;i<load_pins_num;i++){
        Point location(load_pins_x[i]*1./Timing::_db_unit, load_pins_y[i]*1./Timing::_db_unit);  // 假设 Point 有 (x,y) 构造函数
        Inst* inst = new Inst(std::to_string(c++), location);
        inst->cap=load_pins_cap[i];
        inst->ID=i;
        insts.push_back(inst);
    }
    auto lel=LevelSolve();
    std::cout<<"------ start Level ------\n";
    std::vector<Inst*> next_insts =lel.assignApply(insts,max_fanout,max_len,1);
    next_level_load_pins_num = next_insts.size();
    if(next_level_load_pins_x==nullptr)next_level_load_pins_x = new int[next_insts.size()];
    if(next_level_load_pins_y==nullptr)next_level_load_pins_y = new int[next_insts.size()];
    if(next_level_load_pins_cap==nullptr)next_level_load_pins_cap = new double[next_insts.size()];
    if(buffer_cell_masters==nullptr)buffer_cell_masters = new std::string[next_insts.size()];
    if(Timing::Cobufferlibs.size()>0){
            auto cmp = [](std::string lib_1, std::string lib_2) { return CtsLibs::_lib_maps[lib_1]->getDelayIntercept() < CtsLibs::_lib_maps[lib_2]->getDelayIntercept(); };
            std::ranges::sort(Timing::Cobufferlibs, cmp);
    }
    for(int i=0;i<next_insts.size();i++){
        next_level_load_pins_x[i]=Timing::_db_unit*next_insts[i]->_location.x();
        next_level_load_pins_y[i]=Timing::_db_unit*next_insts[i]->_location.y();
        if(Timing::Cobufferlibs.size()>0){
            next_level_load_pins_cap[i]=CtsLibs::_lib_maps[Timing::Cobufferlibs[0]]->get_init_cap();
            buffer_cell_masters[i]=Timing::Cobufferlibs[0];
        }
        else if(Timing::bufferlibs.size()>0){
            next_level_load_pins_cap[i]=CtsLibs::_lib_maps[Timing::bufferlibs[0]]->get_init_cap();
            buffer_cell_masters[i]=Timing::bufferlibs[0];
        }
        else next_level_load_pins_cap[i]=0.;
    }

    next_cluster_num=next_level_load_pins_num;
    Pointc* h_clusters=new Pointc[next_cluster_num];
    Pointc *cpts=new Pointc[load_pins_num];
    cudaMemcpy(cpts,lel.pts,load_pins_num*sizeof(Pointc),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_clusters,lel.clusters,next_cluster_num*sizeof(Pointc),cudaMemcpyDeviceToHost);
    if(next_clusters_segment==nullptr)next_clusters_segment = new int[next_cluster_num+1];
    if(next_load_pins_index==nullptr)next_load_pins_index = new int[load_pins_num];
    next_clusters_segment[0]=0;
    for(int i=0;i<next_cluster_num;i++)next_clusters_segment[i+1]=next_clusters_segment[i]+h_clusters[i].count;
    // for(int i=0;i<load_pins_num;i++)next_load_pins_index[i]=cpts[i].instID,std::cout<<cpts[i].instID<<' ';
    std::cout<<"------ end Level ------\n";
}
void MultiLevelRun(const int load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const int max_fanout, const double max_len, 
    const int fixed_num, const int* fixed_x, const int* fixed_y, int &root_x,int &root_y, std::string& root_cell_master, int &level){
    if(Timing::_db_unit==0)Timing::init();
    int size=load_pins_num;
    level=0;
    std::vector<Inst*>insts;
    int c=0;
    std::cout<<"level "<<level<<" size :"<<size<<'\n';
    for(int i=0;i<load_pins_num;i++){
        Point location(load_pins_x[i]*1./Timing::_db_unit, load_pins_y[i]*1./Timing::_db_unit);  // 假设 Point 有 (x,y) 构造函数
        Inst* inst = new Inst(std::to_string(c++), location);
        inst->cap=load_pins_cap[i];
        inst->ID=i;
        insts.push_back(inst);
    }
    if(Timing::Cobufferlibs.size()>0){
            auto cmp = [](std::string lib_1, std::string lib_2) { return CtsLibs::_lib_maps[lib_1]->getDelayIntercept() < CtsLibs::_lib_maps[lib_2]->getDelayIntercept(); };
            std::ranges::sort(Timing::Cobufferlibs, cmp);
    }
    
    while(size>1){
        auto lel=LevelSolve();
        std::vector<Inst*> next_insts = lel.assignApply(insts,max_fanout,max_len,level);
        for(int i=0;i<next_insts.size();i++){
            if(Timing::Cobufferlibs.size()>0)next_insts[i]->cap=CtsLibs::_lib_maps[Timing::Cobufferlibs[0]]->get_init_cap();
            else if(Timing::bufferlibs.size()>0)next_insts[i]->cap=CtsLibs::_lib_maps[Timing::bufferlibs[0]]->get_init_cap();
            else next_insts[i]->cap=0.001;
        }
        // for(int i=0;i<next_insts.size();i++){
        //     printf("%d %d %f\n",(int)(Timing::_db_unit*next_insts[i]->_location.x()),(int)(Timing::_db_unit*next_insts[i]->_location.y()),next_insts[i]->cap);
        //     // std::cout<<(int)Timing::_db_unit*next_insts[i]->_location.x()<<' '<<(int)Timing::_db_unit*next_insts[i]->_location.y()<<' '<<next_insts[i]->cap<<'\n';
        // }
        insts=next_insts;
        size=next_insts.size();
        level++;
        std::cout<<"level "<<level<<" size :"<<size<<'\n';
    }
    root_x=insts[0]->_location.x()*Timing::_db_unit;
    root_y=insts[0]->_location.y()*Timing::_db_unit;
    // std::cout<<root_x<<" "<<root_y<<'\n';
    if(Timing::Cobufferlibs.size()>0)root_cell_master=Timing::Cobufferlibs[0];
    else if(Timing::bufferlibs.size()>0)root_cell_master=Timing::bufferlibs[0];
    else root_cell_master="";
}
void CBSTreeSingleRun(const int load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const double skew_bound, const int type,
    int& root,  int*& left_child, int*& right_child, int* &tree_node_x, int* & tree_node_y,int &tree_node_num)
{
    if(Timing::_db_unit==0)Timing::init();
    Pointc *cpts=new Pointc[load_pins_num];
    Pointc *pts;
    int cnt=0;
    for(int i=0;i<load_pins_num;i++){
        cpts[i]=Pointc(load_pins_x[i]*1./Timing::_db_unit,load_pins_y[i]*1./Timing::_db_unit);
        cpts[i].cap=load_pins_cap[i];
    }
    cudaMalloc(&pts,load_pins_num*sizeof(Pointc));
    cudaMemcpy(pts, cpts, load_pins_num*sizeof(Pointc), cudaMemcpyHostToDevice);
    Pointc* clusters;
    cudaMalloc(&clusters,sizeof(Pointc));
    Pointc clusters1;
    clusters1.count=load_pins_num;
    cudaMemcpy(clusters,&clusters1,sizeof(Pointc),cudaMemcpyHostToDevice);
    int *clusters_segment;
    cudaMalloc(&clusters_segment,2*sizeof(int));
    int clusters_segment_1[2];
    clusters_segment_1[0]=0,clusters_segment_1[1]=load_pins_num;
    cudaMemcpy(clusters_segment,clusters_segment_1,2*sizeof(int),cudaMemcpyHostToDevice);
    std::vector<Inst*>insts;
    int c=0;
    for(int i=0;i<load_pins_num;i++){
        Point location(load_pins_x[i], load_pins_y[i]);  // 假设 Point 有 (x,y) 构造函数
        Inst* inst = new Inst(std::to_string(c++), location);
        inst->cap=load_pins_cap[i];
        insts.push_back(inst);
    }
    std::cout<<"------ start CBSTree ------\n";
    int t=3;
    if(type==0)t=1;
    auto tmr=TreeMaker(pts,clusters,clusters_segment,insts,insts.size(),1,t);
    if(left_child==nullptr)left_child=new int[insts.size()*3];
    if(right_child==nullptr)right_child=new int[insts.size()*3];
    if(tree_node_x==nullptr)tree_node_x=new int[insts.size()*3];
    if(tree_node_y==nullptr)tree_node_y=new int[insts.size()*3];
    Area *hareas=new Area[3*insts.size()];
    cudaMemcpy(hareas,tmr.areas,3*insts.size()*sizeof(Area),cudaMemcpyDeviceToHost);
    cudaFree(tmr.areas);
    c=0;
    root=PreOrder(hareas,c,insts.size(),0,left_child,right_child,tree_node_x,tree_node_y);
    tree_node_num=c;
    // TreeMaker(pts,insts,load_pins_num);
    std::cout<<"------ end CBSTree ------\n";
}
void CBSTreeRun(const int net_num, const int* load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const double skew_bound, 
    int*& root, int*& left_child, int*& right_child, int* &tree_node_x, int* &tree_node_y,int* &tree_node_num)
{
    Timing::init();
    int* clusters_segment;
    cudaMalloc(&clusters_segment,net_num*sizeof(int));
    int clusters_segment_1[net_num];
    Pointc clusters1[net_num];
    for(int i=0;i<net_num;i++){
        clusters1[i].count=load_pins_num[i];
        if(i==0)clusters_segment_1[i]=0;
        else clusters_segment_1[i]=clusters_segment_1[i-1]+load_pins_num[i-1];
    }
    int pin_num=clusters_segment_1[net_num-1]+load_pins_num[net_num-1];
    cudaMemcpy(clusters_segment,clusters_segment_1,net_num*sizeof(int),cudaMemcpyHostToDevice);
    Pointc *cpts=new Pointc[pin_num];
    Pointc *pts;
    int cnt=0;
    for(int i=0;i<pin_num;i++){
        cpts[i]=Pointc(load_pins_x[i]*1./Timing::_db_unit,load_pins_y[i]*1./Timing::_db_unit);
        cpts[i].cap=load_pins_cap[i];
    }
    cudaMalloc(&pts,pin_num*sizeof(Pointc));
    cudaMemcpy(pts, cpts, pin_num*sizeof(Pointc), cudaMemcpyHostToDevice);
    Pointc* clusters;
    cudaMalloc(&clusters,net_num*sizeof(Pointc));
    cudaMemcpy(clusters,clusters1,net_num*sizeof(Pointc),cudaMemcpyHostToDevice);
    std::vector<Inst*>insts;
    int c=0;
    for(int i=0;i<pin_num;i++){
        Point location(load_pins_x[i], load_pins_y[i]);  // 假设 Point 有 (x,y) 构造函数
        Inst* inst = new Inst(std::to_string(c++), location);
        inst->cap=load_pins_cap[i];
        insts.push_back(inst);
    }
    std::cout<<"------ start CBSTree ------\n";
    auto tmr=TreeMaker(pts,clusters,clusters_segment,insts,insts.size(),net_num);
    if(root==nullptr)root = new int[net_num];
    if(left_child==nullptr)left_child=new int[insts.size()*3];
    if(right_child==nullptr)right_child=new int[insts.size()*3];
    if(tree_node_x==nullptr)tree_node_x=new int[insts.size()*3];
    if(tree_node_y==nullptr)tree_node_y=new int[insts.size()*3];
    if(tree_node_num==nullptr)tree_node_num=new int[net_num+1];    
    Area *hareas=new Area[3*insts.size()];
    cudaMemcpy(hareas,tmr.areas,3*insts.size()*sizeof(Area),cudaMemcpyDeviceToHost);
    cudaFree(tmr.areas);
    c=0;
    tree_node_num[0]=0;
    for(int i=0;i<net_num;i++){
        int pre=c;
        root[i]=PreOrder(hareas,c,pre,insts.size()+i,left_child,right_child,tree_node_x,tree_node_y);
        tree_node_num[i+1]=c;
    }
    std::cout<<"------ end CBSTree ------\n";
}
void TimingInit(const double unit_h_cap, const double unit_h_res, const double unit_v_cap, const double unit_v_res, const int db_unit){
    Timing::init(unit_h_cap, unit_h_res, unit_v_cap, unit_v_res, db_unit);
}
void AddCell(const std::string& cell_master, const int slew_in_index_num, const double* slew_in_index,
    const int cap_out_index_num, const double* cap_out_index, const int delay_mid_value_num, const double* delay_mid_value,
    const int slew_mid_value_num, const double* slew_mid_value, const double init_cap){
    Timing::AddCellLib(cell_master, slew_in_index_num,slew_in_index,
    cap_out_index_num, cap_out_index, delay_mid_value_num, delay_mid_value,
    slew_mid_value_num, slew_mid_value, init_cap);
}
void AddCellCap(const std::string& cell_master, const double init_cap){
    Timing::AddCellLib(cell_master, init_cap);
}
void AddBuffers(int buffer_num, const std::string* cell_masters){
    for(int i=0;i<buffer_num;i++){
        Timing::bufferlibs.push_back(cell_masters[i]);
    }
}
void AddBuffers(const std::string& cell_master, const int slew_in_index_num, const double* slew_in_index,
    const int cap_out_index_num, const double* cap_out_index, const int delay_mid_value_num, const double* delay_mid_value,
    const int slew_mid_value_num, const double* slew_mid_value,const double init_cap){
    Timing::AddCellLib(cell_master, slew_in_index_num,slew_in_index,
    cap_out_index_num, cap_out_index, delay_mid_value_num, delay_mid_value,
    slew_mid_value_num, slew_mid_value, init_cap);
    Timing::Cobufferlibs.push_back(cell_master);
    
}
void SkewRun(const int net_num, const int* load_pins_num, const int* load_pins_x,const int* load_pins_y, const double* load_pins_cap, const double skew_bound, 
    int*& root, int*& left_child, int*& right_child, int* &tree_node_x, int* &tree_node_y,int* &tree_node_num)
{
        if(Timing::_db_unit==0)Timing::init();
        int* clusters_segment;
        cudaMalloc(&clusters_segment,net_num*sizeof(int));
        int clusters_segment_1[net_num];
        Pointc clusters1[net_num];
        for(int i=0;i<net_num;i++){
            clusters1[i].count=load_pins_num[i];
            if(i==0)clusters_segment_1[i]=0;
            else clusters_segment_1[i]=clusters_segment_1[i-1]+load_pins_num[i-1];
        }
        int pin_num=clusters_segment_1[net_num-1]+load_pins_num[net_num-1];
        cudaMemcpy(clusters_segment,clusters_segment_1,net_num*sizeof(int),cudaMemcpyHostToDevice);
        Pointc *cpts=new Pointc[pin_num];
        Pointc *pts;
        int cnt=0;
        for(int i=0;i<pin_num;i++){
            cpts[i]=Pointc(load_pins_x[i]*1./Timing::_db_unit,load_pins_y[i]*1./Timing::_db_unit);
            cpts[i].cap=load_pins_cap[i];
        }
        std::vector<Inst*>insts;
        int c=0;
        for(int i=0;i<pin_num;i++){
            Point location(load_pins_x[i]/Timing::_db_unit, load_pins_y[i]/Timing::_db_unit);  // 假设 Point 有 (x,y) 构造函数
            Inst* inst = new Inst(std::to_string(c++), location);
            inst->cap=load_pins_cap[i];
            insts.push_back(inst);
        }
        cudaMalloc(&pts,pin_num*sizeof(Pointc));
        cudaMemcpy(pts, cpts, pin_num*sizeof(Pointc), cudaMemcpyHostToDevice);
        Pointc* clusters;
        cudaMalloc(&clusters,net_num*sizeof(Pointc));
        cudaMemcpy(clusters,clusters1,net_num*sizeof(Pointc),cudaMemcpyHostToDevice);
        if(left_child==nullptr)left_child=new int[pin_num*3];
        if(right_child==nullptr)right_child=new int[pin_num*3];
        if(tree_node_x==nullptr)tree_node_x=new int[pin_num*3];
        if(tree_node_y==nullptr)tree_node_y=new int[pin_num*3];
        if(root==nullptr)root=new int[pin_num*3];
        if(tree_node_num==nullptr)tree_node_num=new int[net_num+1];  
        auto bst = BoundSkewTree(pts,clusters,clusters_segment,pin_num,net_num);
        if(net_num==1)bst.big=true;
        bst.build(80.,1000,0.,0.,0.,0.);
        bst.GPUtoCPU(pts);
        // std::cout<<"skew\n";
        if(Timing::Cobufferlibs.size()>0){
            auto cmp = [](std::string lib_1, std::string lib_2) { return CtsLibs::_lib_maps[lib_1]->getDelayIntercept() < CtsLibs::_lib_maps[lib_2]->getDelayIntercept(); };
            std::ranges::sort(Timing::Cobufferlibs, cmp);
        }
        for(int i=0;i<net_num;i++){
            if(Timing::Cobufferlibs.size()>0)bst._root_buf_node[i]->_cell_master=Timing::Cobufferlibs[0], bst._root_buf_node[i]->set_cap_load(CtsLibs::_lib_maps[Timing::Cobufferlibs[0]]->get_init_cap());
            // std::cout<<bst._root_buf_node[i]->get_cap_load()<<'\n';
        }
        Timing::checkskew(bst._root_buf_node,insts,bst.load_nodes,net_num);  
        // Area *hareas=new Area[3*pin_num];
        // cudaMemcpy(hareas,bst.areas,3*pin_num*sizeof(Area),cudaMemcpyDeviceToHost);
        // cudaFree(bst.areas);
        // int c=0;
        // for(int i=0;i<net_num;i++){
        //     int pre=c;
        //     root[i]=PreOrder(hareas,c,pre,pin_num+i,left_child,right_child,tree_node_x,tree_node_y);
        //     // std::cout<<c<<'\n';
        //     tree_node_num[i+1]=c;
        // }
}