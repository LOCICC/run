#pragma once
#include "Components.h"
#include "Timing.h"
#include "Instgpu.hh"
#include "Nodegpu.hh"
#include "Enumgpu.hh"
#include <unordered_map>
#include <limits>
#include <math_constants.h>
namespace gcts {
enum class IntersectType
{
  kNone,
  kEndpoint,  // intersect point is endpoint of line segment
  kCrossing,  // intersect point is in both line segment
  kOverlap,   // intersect some part of line segment
  kSame,      // two line are same
};
enum class RelativeType
{
  kLeft,
  kRight,
  kTop,
  kBottom,
  kManhattanParallel,
};
enum class LineType
{
  kVertical,
  kHorizontal,
  kManhattan,
  kFlat,
  kTilt,
};
using PtPair = Pt[2];
class BoundSkewTree
{
 public:
  BoundSkewTree(){};
  BoundSkewTree(int num,const std::optional<double>& skew_bound = std::nullopt);
  BoundSkewTree(Pointc* pts,Pointc* clusters,int* clusters_segment,int pinnum,int net_num);
  ~BoundSkewTree(){
    if(net_len!=nullptr)cudaFree(net_len);
    if(net_delay!=nullptr)cudaFree(net_delay);
  };
  void build(const double skew_bound, const int db_unit, const double unit_h_cap, const double unit_h_res,
  const double unit_v_cap, const double unit_v_res);
  void Init(Pointc* pts,Pointc* clusters,int* clusters_segment,int pinnum,int clusters_num);
  void run();
  void setPinsize(int num){netpinnum[net_num++]=num;pinsize+=num;}
  void biPartition(Region* octagon,Pt* pts,int *ID,Area* bound_areas,int *bound_areas_in,int *bound_areas_sum,int *bound_areas_sum_half);
  void scanInt(int* d_in,int* Intout,int size);
  void scanDouble(double* d_in,double* Intout,int size);
  void coutDouble(double* d,int size);
  void coutInt(int* d,int size);
  void coutAreas(Area* a,int l,int r);
  void coutAreaslr(Area* d_a, int l, int r);
  void coutAreassum(Area* d_a, int l, int r);
  void calculateCap(double* capsum, double* cap);
  void bottomUp();
  int calculateID(int *ID,int *avg_size, int beforeIDsize,int size,int beginID);
  int getlastInt(int* item,int size);
  double getlastDouble(double* item,int size);
  void convexHullsort(Region* octagon,int size);
  void calcOctagonnum(Region* octagon,int *octsum,int size);
  void calcflag(int *flag,int *octsum,int size);
  void initAreas();
  void areaBound(Area* bound_areas,int *bound_areas_in,int *bound_areas_sum,int* bound_areas_sum_half,Region* octagon,int beginID,int areasize);
  void boundAreaSort(Area*result,Area* bound_areas,int *bound_areas_sum,int* bound_areas_sum_half,int * bound_areas_in,int* flag,int areasize);
  void AreaValSort(Area*areas,int k);
  int Areasplit(Area* areas,int l,int r,int size);
  double bound_diameter(int left_num,int kid,int size);
  void findMaxDistanceOptimized(Area* areas, int* d_ifbound,int k, int  left_num,int allsize,int side, double& max_dist);
  void octagonDivide(int *ID,Area* bound_areas,int* bound_areas_in,int* bound_areas_sum,int* bound_areas_sum_half,int beginID,int areasize,int IDadd,int*F,int* areasid,int sum_size,int level);
  void SegmentedRadixSort(Region* octagon,int *octsum, uint64_t *d_keys_in,int *d_values_in,uint64_t *d_keys_out,int *d_values_out,int num_segments,int num_items);
  int getArealeft(Area* areas,int k);
  int getArearight(Area* areas,int k);
  void calculateCenter(int begin,int end);
  void bound_diameter_level(int* left_num,int beginID,int areasize,double *cost,int *F,int level);
  void findMaxDistance(Area* areas, int* d_ifbound,int kid, int  left_num,int allsize,int side, double& max_dist);
  void topDown(int *levelsum);
  void AreaValSortLevel(int*F,int beginID,int areasize);
  void AreaValSortLevelSmall(int* F,int beginID,int areasize);
  void AreaValSortLevelBig(Area* areas1,int beginID,int areasize);
  void calcOctagonTwolevelbig(Area* areas, Region* octagon,int* left_num,double DOUBLE_MIN,double DOUBLE_MAX,int beginID,int areasize);
  void Areasplitlevelbig(Area* areas,int *left_num,int beginID,int areasize,int * areasid,double DOUBLE_MAX);
  void coutUInt(unsigned long* d,int size);
  void setCore(const Point& loc1,const Point& loc2);
  __device__ static double ptToLineDist(Pt& p, const Line& l, Pt& closest);
  __device__ static double ptToLineDistNotManhattan(Pt& p, const Line& l, Pt& closest);
  __device__ static double ptToLineDistManhattan(Pt& p, const Line& l, Pt& closest);
  __device__ static double msDistance(Trr& ms1, Trr& ms2);
  __device__ static void checkMs(Trr& ms);
  __device__ static void makeIntersect(Trr& ms1, Trr& ms2, Trr& intersect);
  __device__ static void coreMidPoint(Trr& ms, Pt& mid);
  __device__ static void lineToMs(Trr& ms, const Pt& p1, const Pt& p2);
  __device__ static void lineToMs(Trr& ms, const Line& l);
  __device__ static IntersectType lineIntersect(Pt& p, Line& l1, Line& l2);
  __device__ static double distance(const Pt& p1, const Pt& p2);
  __device__ static bool Equal_(const double& a, const double& b);
  __device__ static bool Equal(const double& a, const double& b);
  __device__ static bool onLine(Pt& p, const Pt* l);
  __device__ static bool onLine(Pt& p, const Line& l);
  __device__ static double crossProductR(const Pt& p1, const Pt& p2, const Pt& p3);
  __device__ static double distanceR(const Pt& p1, const Pt& p2);
  __device__ static bool isSegmentTrr(const Trr& trr);
  __device__ static void msToLine(Trr& ms, Pt& p1, Pt& p2);
  __device__ static void msToLine(Trr& ms, Line& l);
  __device__ static bool isSame(const Pt& p1, const Pt& p2);
  __device__ static void trrCore(const Trr& trr, Trr& core);
  __device__ static bool boundBoxOverlap(const double& x1, const double& y1, const double& x2, const double& y2, const double& x3, const double& y3,
                               const double& x4, const double& y4, const double& epsilon);
  __device__ static bool boundBoxOverlap(const Line& l1, const Line& l2, const double& epsilon);
  __device__ static bool inBoundBox(const Pt& p, const Line& l);
  __device__ static void trrToPt(const Trr& trr, Pt& pt);
  __device__ static void trrToRegion(Trr& trr, Pt* region,int &cnt);
  __device__ static double lineDist(Line& l1, Line& l2, PtPair& closest);
  __device__ static LineType lineType(const Pt& p1, const Pt& p2);
  __device__ static LineType lineType(const Line& l);
  __device__ static bool isParallel(const Line& l1, const Line& l2);
  __device__ static void calcCoord(Pt& p, const Line& l, const double& shift);

  void recursiveBottomUp();
  void initms(Side<Trr>*& ms,int num);
  void initSidePts(Side<Pts>*& join,Pt *&ptsl,Pt*&ptsr,int num,int cnt);

  void embedding();
  void BoundSkewTreeAddNet(const std::string& net_name, const std::vector<Pin*>& pins);
  void convert();
  void calcNetLen();
  void estimateNetDelay();
  void SaltConvert(Area* _root,Area* h_areas,int net_id);
  void GPUtoCPU(Pointc* pts);
  void CPUtoGPU(Pointc* pts);
  void setArealr(int id);
  void runTwo(int id,bool estimation,double _skew_bound);
  void initMrAndConv(int cnt);
  Area* areas;
  bool big=false;
  std::vector<Area*> _unmerged_nodes;
  std::unordered_map<int, Node*> _node_map;
  std::vector<Area>H_areas;
  int pinsize=0;
  int maxlevel=0;
  int *levelsum=nullptr;
  double cap;
  int net_num=0;
  int *netpinnum;
  int clusterflag=0;
  double* net_len=nullptr;
  double* net_delay=nullptr;
  std::vector<Inst*> _root_buf;
  std::vector<Node*> _root_buf_node;
  std::vector<Area*> _root;
  std::vector<Node*> load_nodes;
  int pinid=0;
  double g_skew_bound=80;
  std::unordered_map<std::string, int> map;
};
}