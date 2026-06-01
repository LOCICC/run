#pragma once
#include "Instgpu.hh"
#include "BoundSkewTree.h"
#include "GPUClustering.h"
#include "TreeMaker.h"
#include <cmath>  
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <unordered_map>
#include <limits>
#include "Pingpu.hh"
namespace gcts {  

class LevelSolve
{
    public:
    LevelSolve(){}
    ~LevelSolve(){
        cudaFree(pts);
        cudaFree(clusters);
        cudaFree(clusters_segment);
    }
    static void guideCenter(Pointc* pts,Pointc* clusters,Pointc* guide_locs,int* clusters_segment,std::vector<Inst*>& insts,int instsize,int cluster_num);
    static void LCA(Area* areas,Pointc* pts,Pointc* clusters,int *levelsum,int *clusters_segment,int *ancestors,int maxlevel,Pointc* guide_locs,int instsize,int cluster_num);
    static void levelProcess(Pointc *guide_locs,Pointc *pts, Pointc* clusters,int *clusters_segment,std::vector<Inst*>& insts,int cluster_num,int level, const bool& shift);
    std::vector<Inst*> assignApply(std::vector<Inst*>& insts,int max_fanout,double max_len,int level);
    static void coutPointc(Pointc* d,int size);
    static void coutDouble(double* d,int size);
    static void coutInt(int* d,int size);
    void CPUtoGPU(std::vector<Inst*>& insts);
    void GPUtoCPU(std::vector<Inst*>& insts,Pointc* guide_locs,int cluster_num);
    double h_getUnitCap=7.595e-05;
    double h_getUnitRes=7.83333e-05;
    double h_getDbUnit=1.;
    int instsize;
    Pointc* pts;
    Pointc* clusters;
    int *clusters_segment;
};
}