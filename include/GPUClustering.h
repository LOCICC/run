#pragma once
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/random.h>
#include <unordered_map>
#include <limits>
#include "Instgpu.hh"
#include "Pingpu.hh"
namespace gcts {  
struct segment{
    int l,r;
};
class GPUClustering
{
    public:
    GPUClustering(){}
    void INIT(int num);
    static void calculatesum(Pointc* pts, Pointc* new_pts,int*assignments,int k,int num_instances);
    // std::vector<std::vector<Inst*>> kMeansPlusBig(Pointc* pts,Pointc*cpts,const size_t& k, const int& seed,const size_t& max_iter, const size_t& no_change_stop,int num_instances);
    static void kMeansPlusBig(Pointc* clusters,Pointc* pts,const size_t& k, const int& seed,const size_t& max_iter, const size_t& no_change_stop,int num_instances);
    std::vector<std::vector<Inst*>> kMeansPlus(const std::vector<Inst*>& insts,const size_t& k, const int& seed=0,const size_t& max_iter=5, const size_t& no_change_stop=5);
    std::vector<std::vector<Inst*>> iterClusteringcpu(const std::vector<Inst*>& insts, const size_t& max_fanout,
                                                        const size_t& iters = 100, const size_t& no_change_stop = 5,
                                                        const double& limit_ratio = 0.8, const bool& log = false);
    static std::vector<std::vector<Inst*>> kMeans(const std::vector<Inst*>& insts, const size_t& k, const int& seed,
                                                          const size_t& max_iter);
    // void find_argmax_cub(double* d_array, cub::KeyValuePair<int, int> *d_argmax, int size) ;
    static void calcVariance(double* d_data,double* variance, int n);
    std::vector<Point> getCentroidBufferscpu(const std::vector<std::vector<Inst*>>& clusters);
    static void calcBalanceVariance(Pointc* pts,Pointc* clusters,int* clusters_segment,int cluster_num,std::vector<Inst*>& insts,double* V,
                                    const double& cap_coef = 1.0, const double& delay_coef = 1.0);
    static void calcDelayVariance(Pointc* clusters,int* clusters_segment,int cluster_num,double* net_len,double* net_delay,double* cluster_cap,double *V);
    static void calcCapVariance(Pointc* clusters,int* clusters_segment,int cluster_num,double* net_len,double* cluster_cap,double *V);
    static int iterClustering(std::vector<Inst*>& insts,Pointc*pts,Pointc* clusters, const size_t& max_fanout,
                           const size_t& iters, const size_t& no_change_stop,const double& limit_ratio = 0.8);
    static void initClusterSegment(Pointc* clusters,int *clusters_segment,int num);
    static int selectRandom(double *distances,thrust::default_random_engine &gen,int num_instances,int seed);
    static void Auction(const std::vector<Inst*>& insts,Pointc* pts,Pointc* clusters,Pointc*buffers,int *clusters_segment,int max_fanout,int cluster_num);
    static void SimpleMCF(Pointc* pts,Pointc* clusters,Pointc* buffers,int *buffers_segment,int max_fanout,int size);
    static void AuctionRun(Pointc* pts,Pointc* clusters,Pointc* buffers,int *buffers_segment,double * cost_matrices,int max_fanout,int size);
    static void AuctionToCluster(Pointc* pts,Pointc* clusters,Pointc* buffers,int *clusters_segment,int *assignments,int max_fanout,int num_instances);
    static int slackClustering(Pointc* pts,Pointc*clusters,int *clusters_segment,std::vector<Inst*>& insts,double max_len,int k);
    static void kMeansPlusSmall(Pointc* pts,Pointc* clusters,Pointc* best_clusters,int* clusters_segment,int* best_clusters_segment,int *k_segment,int *k,int size,int size1,int num_instances);
    static void CheckNetLength(Pointc *pts,Pointc* clusters,int *clusters_segment,int *flag,std::vector<Inst*>& insts,double max_len,int cluster_num);
    static void selectNet(Pointc *pts,Pointc *select_pts,Pointc* clusters,Pointc* select_clusters,int *clusters_segment,
    int *select_clusters_segment,int* clusterid,int *flag,int num_instances,int &select_num_instances,int cluster_num, int &select_cluster_num);
    static int removeEmpty(Pointc*clusters,int size);
    static void coutPointc(Pointc* d,int size);
    static void coutDouble(double* d,int size);
    static void coutInt(int* d,int size);
    static void coutCost(Pointc*h_pts, int* clusters_segment, Pointc* buffers,int cluster_num,int size);
    static void AllocateRun(std::vector<Inst*>& insts,Pointc* pts,Pointc* clusters,Pointc*buffers,int max_fanout,int cluster_num);
    double h_getUnitCap=7.595e-05;
    double h_getUnitRes=7.83333e-05;
    double h_getDbUnit=1.;
    double h_getMinNetlen=300;
    int instsize;
};
}