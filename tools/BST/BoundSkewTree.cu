#include <stack>
#include <cub/cub.cuh>
// #include "GeomCalc.cuh"
#include "BoundSkewTree.h"
#include <thrust/device_vector.h> 
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <chrono>
namespace gcts {

__device__ double BoundSkewTree::distance(const Pt& p1, const Pt& p2){
  return fabs(p1.x - p2.x) + fabs(p1.y - p2.y);
}
__device__ bool BoundSkewTree::Equal_(const double& a, const double& b){
  return fabs(a - b) < 1e-7;
}
int BoundSkewTree::getlastInt(int* item,int size){
    int last_element;
    // 仅复制最后一个元素（偏移量 size-1）
    cudaMemcpy(&last_element, &item[size-1], sizeof(int), cudaMemcpyDeviceToHost);
    return last_element;
}
__device__ double crossProduct(const Pt& p1, const Pt& p2, const Pt& p3){
  return (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y);
}
__device__ Pt centerPt(Region& Pts){
  Pt center;
  if (Pts.size==0) {
    center.x=0.,center.y=0.;
    return center;
  }
  double x = 0, y = 0;
  for(int i=0;i<Pts.size;i++){
    x += Pts.pts[i].x;
    y += Pts.pts[i].y;
  }
  center.x=x / Pts.size,center.y=y / Pts.size;
  return center;
}
__device__ bool BoundSkewTree::onLine(Pt& p, const Pt* l){
  int kHead=0,kTail=1;
  auto len = distance(l[kHead], l[kTail]);
  auto len_to_head = distance(p, l[kHead]);
  auto len_to_tail = distance(p, l[kTail]);
  if (std::abs(len_to_head + len_to_tail - len) < 2 * kEpsilon) {
    if (Equal_(len_to_head, 0)) {
      p = l[kHead];
      return true;
    } else if (Equal_(len_to_tail, 0)) {
      p = l[kTail];
      return true;
    } else {
      auto delta_x = std::abs(l[kTail].x - l[kHead].x);
      auto delta_y = std::abs(l[kTail].y - l[kHead].y);
      if (delta_y > delta_x) {
        auto temp_x = (l[kTail].x - l[kHead].x) * (p.y - l[kHead].y) / (l[kTail].y - l[kHead].y) + l[kHead].x;
        if (Equal_(temp_x, p.x)) {
          p.x = temp_x;
          return true;
        }
      } else {
        auto temp_y = (l[kTail].y - l[kHead].y) * (p.x - l[kHead].x) / (l[kTail].x - l[kHead].x) + l[kHead].y;
        if (Equal_(temp_y, p.y)) {
          p.y = temp_y;
          return true;
        }
      }
    }
  }
  return false;
}
__device__ void device_convexHull(Region& octagon){
  // check pts num
  int num=octagon.size;
  if (num < 2) {
    return;
  }
  if (num == 2) {
    auto dist = BoundSkewTree::distance(octagon.pts[0], octagon.pts[1]);
    if (BoundSkewTree::Equal_(dist, 0)) {
      octagon.size=num-1;
    }
    return;
  }
  // calculate convex hull by Andrew algorithm
  // std::ranges::sort(pts, [](const Pt& p1, const Pt& p2) { return p1.x + kEpsilon < p2.x || (Equal(p1.x, p2.x) && p1.y < p2.y); });
  // quickSort(octagon->pts,0,num-1);
  Pt ans[16]; 
  size_t k = 0;
  for (size_t i = 0; i < octagon.size; ++i) {
    while (k > 1 && crossProduct(ans[k - 2], ans[k - 1], octagon.pts[i]) <= kEpsilon) {
      --k;
    }
    ans[k++] = octagon.pts[i];
  }
  for (size_t i = octagon.size - 1, t = k + 1; i > 0; --i) {
    while (k >= t && crossProduct(ans[k - 2], ans[k - 1], octagon.pts[i - 1]) <= kEpsilon) {
      --k;
    }
    ans[k++] = octagon.pts[i-1];
  }
  for (size_t i = 0; i < k - 1; ++i) {octagon.pts[i] = ans[i];}
  octagon.size=k-1;
  // if(id==0)
  // {
  //   printf("%d,%d\n",id, octagon[id].size);
  //   for (size_t i = 0; i <  octagon[id].size; ++i) {printf("oct:%d,%f,%f\n",id,octagon[id].pts[i].x,octagon[id].pts[i].y);}
  // }
}
__global__ void areasort(Area* areas,int* F,int beginID,int areasize){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id>=areasize)return ;
    int l=areas[id+beginID].l;
    int r=areas[id+beginID].r;
    int size=r-l+1;
    if(size<=2)return;
    if(F[l]==-1)return;
    for (size_t i = l; i <= r; ++i) {
    for (size_t j = i + 1; j <=r; ++j) {
        if (areas[j].val < areas[i].val) {
            // std::swap(result[i], result[j]);
            Area temp=areas[j];
            areas[j]=areas[i];
            areas[i]=temp;
        }
    }
}
}
__global__ void boundAreasort(Area* areas,int *bound_areas_sum,int areasize){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id>=areasize)return ;
    int l=bound_areas_sum[id];
    int r=bound_areas_sum[id+1]-1;
    int size=r-l+1;
    for (size_t i = l; i <= r; ++i) {
    for (size_t j = i + 1; j <=r; ++j) {
        if (areas[j].val < areas[i].val) {
            // std::swap(result[i], result[j]);
            Area temp=areas[j];
            areas[j]=areas[i];
            areas[i]=temp;
        }
    }
  }
}
__global__ void initID(int *in,int *d_in,Area*areas,Pt*pts1,Pt* pts2,int *avg_size,int beforeIDsize, int beginID) {//beforeIDsize前一层的树节点数量
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id>beforeIDsize)return ;
    if(id==beforeIDsize){
      d_in[id]=0;
      return ;
    }
    int size=areas[id+beginID].r-areas[id+beginID].l+1;
    areas[id+beginID].size=size;
    if(size>1)avg_size[id]=size;
    else avg_size[id]=0;
    areas[id+beginID].convex_hull.pts=&pts1[16*id];
    areas[id+beginID].convex_hull.size=0;
    areas[id+beginID].mr.pts=&pts2[16*id];
    areas[id+beginID].mr.size=0;

    in[id]=size;
    // printf("%d:(%d,%d)\n",id+beginID,areas[id+beginID].l,areas[id+beginID].r);
    if(size>=2)d_in[id]=2;
    else {
      d_in[id]=0;
      areas[id+beginID].convex_hull.size=1;
      // areas[id+beginID].convex_hull.pts[0].x=areas[areas[id+beginID].l].x;
      // areas[id+beginID].convex_hull.pts[0].y=areas[areas[id+beginID].l].y;
      areas[id+beginID].convex_hull.pts[0]=Pt(areas[areas[id+beginID].l].x,areas[areas[id+beginID].l].y);

      areas[id+beginID].mr.size=1;
      // areas[id+beginID].mr.pts[0].x=areas[areas[id+beginID].l].x;
      // areas[id+beginID].mr.pts[0].y=areas[areas[id+beginID].l].y;
      areas[id+beginID].mr.pts[0]=Pt(areas[areas[id+beginID].l].x,areas[areas[id+beginID].l].y);
      areas[id+beginID].cap=areas[areas[id+beginID].l].cap;
      areas[id+beginID].location=areas[areas[id+beginID].l].location;
      // printf("id+beginID:%d",id+beginID);
    }
    if(areas[id+beginID].p==-1&&size==1){ //一个节点
      areas[id+beginID].left=-1;
      areas[id+beginID].right=-1;
    }
} 
__global__ void copylevel(Area*areas,Area*areas1,int *F,int beginID,int pinsize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=pinsize)return ;
  if(F[id]==-1)return;
  // printf("%d %d,%d,%d\n",id,F[id]+beginID,areas[F[id]+beginID].l,areas[F[id]+beginID].r);
  if((areas[F[id]+beginID].r-areas[F[id]+beginID].l+1)<=2)return;
  areas[id]=areas1[id];
  // printf("copylevel %d\n",areas[id].ptsid);
}
__device__ bool ComparePt(const Pt& p1, const Pt& p2) {
    if (p1.x + kEpsilon < p2.x) return true;
    if (BoundSkewTree::Equal_(p1.x, p2.x) && p1.y < p2.y) return true;
    return false;
}
__global__ void octagonsort(Region *octagon,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=size)return;
  int n = octagon[id].size;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (!ComparePt(octagon[id].pts[j], octagon[id].pts[j + 1])) {
                // 交换元素
                Pt temp = octagon[id].pts[j];
                octagon[id].pts[j] = octagon[id].pts[j + 1];
                octagon[id].pts[j + 1] = temp;
            }
        }
    }
}
__device__ void device_octagonsort(Region &octagon){
  int n = octagon.size;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (!ComparePt(octagon.pts[j], octagon.pts[j + 1])) {
                // 交换元素
                Pt temp = octagon.pts[j];
                octagon.pts[j] = octagon.pts[j + 1];
                octagon.pts[j + 1] = temp;
            }
        }
    }
}
__global__ void initareasort(Area*area1,Area*areas,int*area_sum,int*F,int beginID,int pinsize) {//beforeIDsize前一层的树节点数量
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id>=pinsize)return ;
    if(F[id]==-1)return;
    int offset=id-areas[F[id]+beginID].l;
    // printf("offset:%d %d\n",offset,area_sum[F[id]]);
    area1[area_sum[F[id]]+offset]=areas[id];
} 
__global__ void costcal(Area*areas,double*cost,double*max_dist,int beginID,int areasize,double DOUBLE_MAX) {//beforeIDsize前一层的树节点数量
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id>=areasize)return;
    int size=areas[id+beginID].r-areas[id+beginID].l+1;
    if(size<=2){
      cost[id]=DOUBLE_MAX;
      return ;
    }
    cost[id]=max_dist[id*2]+max_dist[id*2+1];
} 
__global__ void costUpdateBigareas(Area*areas,Area* areas1,int *areasid,int*F,int beginID,int pinsize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=pinsize)return;
  int ida=areasid[id];
  if(id<areas[ida+beginID].l||id>areas[ida+beginID].r)return ;
  if(F[ida])areas1[id]=areas[id];
}
__global__ void costUpdateBig(Area*areas,Area* areas1,int *divide,double*cost,double*min_cost,int i,int *host_bound_areas,int *left_num,int beginID,int*F,int areasize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=areasize)return;
  F[id]=0;
  int l=areas[id+beginID].l;
  int r=areas[id+beginID].r;
  int size=r-l+1;
  if(size<=1){
      divide[id]=-1;
      return ;
  }
  if(size==2){
    divide[id]=1;
    return ;
  }
  if(i>=host_bound_areas[id])return;
  if (cost[id] < min_cost[id]) {
      min_cost[id] = cost[id];
      divide[id]=left_num[id];
      F[id]=1;
  }
}
__global__ void costUpdate(Area*areas,Area* areas1,int *divide,double*cost,double*min_cost,int i,int *host_bound_areas,int *left_num,int beginID,int areasize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=areasize)return;
  int l=areas[id+beginID].l;
  int r=areas[id+beginID].r;
  int size=r-l+1;
  if(size<=1){
      divide[id]=-1;
      return ;
  }
  if(size==2){
    divide[id]=1;
    return ;
  }
  if(i>=host_bound_areas[id])return;
  if (cost[id] < min_cost[id]) {
      min_cost[id] = cost[id];
      divide[id]=left_num[id];
      for(int j=l;j<=r;j++){
        areas1[j]=areas[j];
      }
  }
}
__global__ void initconvex(Area* areas,Pt* pt,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) return;
  areas[id].convex_hull.size=0;
  areas[id].convex_hull.pts=&pt[16*id];
  if(areas[id].l==areas[id].r){
    areas[id].convex_hull.size=1;
    areas[id].convex_hull.pts[0].x=areas[id].x;
    areas[id].convex_hull.pts[0].y=areas[id].y;
  }
}
int BoundSkewTree::calculateID(int *ID,int *avg_size, int beforeIDsize,int size,int beginID){
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    int *d_in,*in;                            // 设备端输入指针
    cudaMalloc(&d_in, (beforeIDsize+1) * sizeof(int));                       
    cudaMalloc(&in, (beforeIDsize) * sizeof(int));
    Pt* pts1,*pts2;
    cudaMalloc(&pts1, 16*(beforeIDsize) * sizeof(Pt));
    cudaMalloc(&pts2, 16*(beforeIDsize) * sizeof(Pt));
    gridSize = (beforeIDsize+1 + blockSize - 1) / blockSize;
    initID<<<gridSize, blockSize>>>(in,d_in,areas,pts1,pts2,avg_size,beforeIDsize,beginID);
    cudaDeviceSynchronize();
    // std::cout<<"calculateID:"<<beginID<<' '<<beforeIDsize<<'\n'<<"ID";
    // coutInt(d_in,beforeIDsize+1);
    // std::cout<<"size";
    // coutInt(in,beforeIDsize);
    scanInt(d_in,ID,beforeIDsize+1);
    //前缀和计算octsum
    int l=getlastInt(ID,beforeIDsize+1);
    // cudaFree(d_temp_storage);
    // cudaFree(d_in);
    return l; 
}
__global__ void setlocation(Area* areas,int beginID,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) return;
  double x=0,y=0;
  for(int i=areas[id+beginID].l;i<=areas[id+beginID].r;i++){
    x+=areas[i].x;
    y+=areas[i].y;
  }
  int lrsize=areas[id+beginID].r-areas[id+beginID].l+1;
  areas[id+beginID].x=x/lrsize;
  areas[id+beginID].y=y/lrsize;
}
__global__ void initOctagon(Region* octagon,Pt* pts,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) return;
  octagon[id].size=0;
  octagon[id].pts=&pts[16*id];
}
__global__ void initareasum(Area*areas,int*area_in,int beginID,int areasize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id > areasize) return;
  if(id==areasize){
    area_in[id]=0;
    return;
  }
  area_in[id]=areas[id+beginID].r-areas[id+beginID].l+1;
}
__global__ void initroot(Area*areas,Pointc*clusters,int *clusters_segment,int pinsize,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) return;
  areas[pinsize+id].l=clusters_segment[id];
  areas[pinsize+id].r=clusters_segment[id]+clusters[id].count-1;
  areas[pinsize+id].size=clusters[id].count;
  areas[pinsize+id].p=-1;
  if(clusters[id].count==0)areas[pinsize+id].l=areas[pinsize+id].r;
  if(clusters[id].count==1)areas[areas[pinsize+id].l].p=pinsize+id;
}
__global__ void initroot(Area*areas,int *netpinnum,int *netpinsum,int pinsize,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) return;
  areas[pinsize+id].l=netpinsum[id];
  areas[pinsize+id].r=netpinsum[id]+netpinnum[id]-1;
  areas[pinsize+id].size=netpinnum[id];
  areas[pinsize+id].p=-1;
  if(netpinnum[id]==1)areas[areas[pinsize+id].l].p=pinsize+id;
}
__global__ void initareas(Area*areas,double* x,double* y,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) return;
  areas[id].l=id;
  areas[id].r=id;
  areas[id].id=id;
  areas[id].cap=0.003;
  areas[id].size=1;
  areas[id].x=x[id];
  areas[id].y=y[id];
  areas[id].location=Pt(x[id],y[id]);
  areas[id].left=-1;
  areas[id].right=-1;
}
__global__ void initareas(Area*areas,Pointc* pts,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) return;
  areas[id].l=id;
  areas[id].r=id;
  areas[id].id=id;
  areas[id].size=1;
  areas[id].x=pts[id].x;
  areas[id].y=pts[id].y;
  areas[id].cap=pts[id].cap;
  areas[id].location=Pt(pts[id].x,pts[id].y);
  areas[id].ptsid=id;
  areas[id].left=-1;
  areas[id].right=-1;
}
__global__ void calcOctagon(Area* areas, Region* octagon,int beginID,int beforeIDsize,double DOUBLE_MIN,double DOUBLE_MAX,int*F) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=beforeIDsize||areas[id+beginID].size<=0)return;
  auto x_p = DOUBLE_MIN, y_p = DOUBLE_MIN, ymx_p = DOUBLE_MIN,
       ypx_p = DOUBLE_MIN;
  auto x_m = DOUBLE_MAX, y_m = DOUBLE_MAX, ymx_m = DOUBLE_MAX,
       ypx_m = DOUBLE_MAX;
  for(int i=areas[id+beginID].l;i<=areas[id+beginID].r;i++){
    F[i]=id;
    auto x=areas[i].x;
    auto y=areas[i].y;
    x_p = fmax(x, x_p);  // 使用 fmax 替代 std::max
    x_m = fmin(x, x_m);  // 使用 fmin 替代 std::min
    y_p = fmax(y, y_p);
    y_m = fmin(y, y_m);
    ymx_p = fmax(y - x, ymx_p);
    ymx_m = fmin(y - x, ymx_m);
    ypx_p = fmax(y + x, ypx_p);
    ypx_m = fmin(y + x, ypx_m);
  }
  octagon[id].pts[0].x=y_p - ymx_p,octagon[id].pts[0].y=y_p;
  octagon[id].pts[1].x=ypx_p - y_p,octagon[id].pts[1].y=y_p;
  octagon[id].pts[2].x=x_p,octagon[id].pts[2].y=ypx_p - x_p;
  octagon[id].pts[3].x=x_p,octagon[id].pts[3].y=x_p + ymx_m;
  octagon[id].pts[4].x=y_m - ymx_m,octagon[id].pts[4].y=y_m;
  octagon[id].pts[5].x=ypx_m - y_m,octagon[id].pts[5].y=y_m;
  octagon[id].pts[6].x=x_m,octagon[id].pts[6].y=ypx_m - x_m;
  octagon[id].pts[7].x=x_m,octagon[id].pts[7].y=x_m + ymx_p;
  octagon[id].size=8;
  // if(id==1)printf("%d,%d\n",areas[id+beginID].l,areas[id+beginID].r);
  // printf("%d,%d\n",areas[id+beginID].l,areas[id+beginID].r);
  // for(int i=areas[id+beginID].l;i<=areas[id+beginID].r;i++){printf("%f,%f\n",areas[i].x,areas[i].y);}
  // for(int i=0;i<8;i++)printf("calcOctagonBefore: %d:%f,%f\n",i,octagon[id].pts[i].x,octagon[id].pts[i].y);
  // device_octagonsort(octagon[id]);
  // device_convexHull(octagon[id]);
}
__global__ void calcOctagonTwo(Area* areas, Region* octagon,int kid,int left_num,double DOUBLE_MIN,double DOUBLE_MAX) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=2)return;
  auto x_p = DOUBLE_MIN, y_p = DOUBLE_MIN, ymx_p = DOUBLE_MIN,
       ypx_p = DOUBLE_MIN;
  auto x_m = DOUBLE_MAX, y_m = DOUBLE_MAX, ymx_m = DOUBLE_MAX,
       ypx_m = DOUBLE_MAX;
  int l,r;
  if(id==0){
    l=areas[kid].l;
    r=l+left_num-1;
  }
  else{
    l=areas[kid].l+left_num;
    r=areas[kid].r;
  }
  for(int i=l;i<=r;i++){
    auto x=areas[i].x;
    auto y=areas[i].y;
    x_p = fmax(x, x_p);  // 使用 fmax 替代 std::max
    x_m = fmin(x, x_m);  // 使用 fmin 替代 std::min
    y_p = fmax(y, y_p);
    y_m = fmin(y, y_m);
    ymx_p = fmax(y - x, ymx_p);
    ymx_m = fmin(y - x, ymx_m);
    ypx_p = fmax(y + x, ypx_p);
    ypx_m = fmin(y + x, ypx_m);
  }
  octagon[id].pts[0].x=y_p - ymx_p,octagon[id].pts[0].y=y_p;
  octagon[id].pts[1].x=ypx_p - y_p,octagon[id].pts[1].y=y_p;
  octagon[id].pts[2].x=x_p,octagon[id].pts[2].y=ypx_p - x_p;
  octagon[id].pts[3].x=x_p,octagon[id].pts[3].y=x_p + ymx_m;
  octagon[id].pts[4].x=y_m - ymx_m,octagon[id].pts[4].y=y_m;
  octagon[id].pts[5].x=ypx_m - y_m,octagon[id].pts[5].y=y_m;
  octagon[id].pts[6].x=x_m,octagon[id].pts[6].y=ypx_m - x_m;
  octagon[id].pts[7].x=x_m,octagon[id].pts[7].y=x_m + ymx_p;
  octagon[id].size=8;
  // if(id==0){
    // for(int i=l;i<=r;i++){printf("%f,%f\n",areas[i].x,areas[i].y);}
    // printf("-------------%d,%d\n",l,r);
    // for(int i=0;i<8;i++)printf("%d:%f,%f\n",i,octagon[id].pts[i].x,octagon[id].pts[i].y);}
}
__device__ double atomicMax(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                       __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double atomicMin(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                       __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}
__global__ void initVals(Area* areas,Val* vals,int pinsize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=pinsize)return;
  vals[id].id=id;
  vals[id].val=areas[id].val;
}
__global__ void setVals(Area* areas,Area*areas1,Val* vals,int pinsize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=pinsize)return;
  areas1[id]=areas[vals[id].id];
}
__global__ void calcOctagonTwolevelbigstep(Area* areas,Region* octagon,int* left_num,double DOUBLE_MIN,double DOUBLE_MAX,int beginID,int id){
  int l,r;
  if((id&1)==0){
    l=areas[id/2+beginID].l;
    r=l+left_num[id/2]-1;
  }
  else{
    l=areas[id/2+beginID].l+left_num[id/2];
    r=areas[id/2+beginID].r;
  }
  // 假设 block_size 是你的线程块大小
  const int block_size = 256;
  __shared__ double c_x_p,c_x_m,c_y_p,c_y_m,c_ymx_p,c_ymx_m,c_ypx_p,c_ypx_m;

  auto x_p = DOUBLE_MIN, y_p = DOUBLE_MIN, ymx_p = DOUBLE_MIN,
       ypx_p = DOUBLE_MIN;
  auto x_m = DOUBLE_MAX, y_m = DOUBLE_MAX, ymx_m = DOUBLE_MAX,
       ypx_m = DOUBLE_MAX;
  // 每个线程处理多个元素
  for (int i = l + threadIdx.x; i <= r; i += blockDim.x) {
      auto x = areas[i].x;
      auto y = areas[i].y;

      x_p = fmax(x, x_p);  // 使用 fmax 替代 std::max 
      x_m = fmin(x, x_m);  // 使用 fmin 替代 std::min
      y_p = fmax(y, y_p);
      y_m = fmin(y, y_m);
      ymx_p = fmax(y - x, ymx_p);
      ymx_m = fmin(y - x, ymx_m);
      ypx_p = fmax(y + x, ypx_p);
      ypx_m = fmin(y + x, ypx_m);
  }
  if(threadIdx.x==0){
    c_x_p = DOUBLE_MIN, c_y_p = DOUBLE_MIN, c_ymx_p = DOUBLE_MIN,
       c_ypx_p = DOUBLE_MIN;
    c_x_m = DOUBLE_MAX, c_y_m = DOUBLE_MAX, c_ymx_m = DOUBLE_MAX,
       c_ypx_m = DOUBLE_MAX;
  }
  __syncthreads();
  atomicMax(&c_x_p, x_p);
  atomicMin(&c_x_m, x_m);
  atomicMax(&c_y_p, y_p);
  atomicMin(&c_y_m, y_m);
  atomicMax(&c_ymx_p, ymx_p);
  atomicMin(&c_ymx_m, ymx_m);
  atomicMax(&c_ypx_p, ypx_p);
  atomicMin(&c_ypx_m, ypx_m);
  __syncthreads();
  if(threadIdx.x==0){
    octagon[id].pts[0].x=c_y_p - c_ymx_p,octagon[id].pts[0].y=c_y_p;
    octagon[id].pts[1].x=c_ypx_p - c_y_p,octagon[id].pts[1].y=c_y_p;
    octagon[id].pts[2].x=c_x_p,octagon[id].pts[2].y=c_ypx_p - c_x_p;
    octagon[id].pts[3].x=c_x_p,octagon[id].pts[3].y=c_x_p + c_ymx_m;
    octagon[id].pts[4].x=c_y_m - c_ymx_m,octagon[id].pts[4].y=c_y_m;
    octagon[id].pts[5].x=c_ypx_m - c_y_m,octagon[id].pts[5].y=c_y_m;
    octagon[id].pts[6].x=c_x_m,octagon[id].pts[6].y=c_ypx_m - c_x_m;
    octagon[id].pts[7].x=c_x_m,octagon[id].pts[7].y=c_x_m + c_ymx_p;
    octagon[id].size=8;
  }
}
__global__ void calcOctagonTwolevel(Area* areas, Region* octagon,int* left_num,double DOUBLE_MIN,double DOUBLE_MAX,int beginID,int areasize) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=areasize)return;
  auto x_p = DOUBLE_MIN, y_p = DOUBLE_MIN, ymx_p = DOUBLE_MIN,
       ypx_p = DOUBLE_MIN;
  auto x_m = DOUBLE_MAX, y_m = DOUBLE_MAX, ymx_m = DOUBLE_MAX,
       ypx_m = DOUBLE_MAX;
  int l,r;
  // printf("%d %d %d %d %d\n",areas[id/2+beginID].l,areas[id/2+beginID].r,id,id/2+beginID,left_num[id/2]);//bug
  if((id&1)==0){
    l=areas[id/2+beginID].l;
    r=l+left_num[id/2]-1;
  }
  else{
    l=areas[id/2+beginID].l+left_num[id/2];
    r=areas[id/2+beginID].r;
  }
  // printf("calcOctagonTwolevel:%d %d %d\n",id,l,r);
  for(int i=l;i<=r;i++){
    auto x=areas[i].x;
    auto y=areas[i].y;
    x_p = fmax(x, x_p);  // 使用 fmax 替代 std::max 
    x_m = fmin(x, x_m);  // 使用 fmin 替代 std::min
    y_p = fmax(y, y_p);
    y_m = fmin(y, y_m);
    ymx_p = fmax(y - x, ymx_p);
    ymx_m = fmin(y - x, ymx_m);
    ypx_p = fmax(y + x, ypx_p);
    ypx_m = fmin(y + x, ypx_m);
  }
  octagon[id].pts[0].x=y_p - ymx_p,octagon[id].pts[0].y=y_p;
  octagon[id].pts[1].x=ypx_p - y_p,octagon[id].pts[1].y=y_p;
  octagon[id].pts[2].x=x_p,octagon[id].pts[2].y=ypx_p - x_p;
  octagon[id].pts[3].x=x_p,octagon[id].pts[3].y=x_p + ymx_m;
  octagon[id].pts[4].x=y_m - ymx_m,octagon[id].pts[4].y=y_m;
  octagon[id].pts[5].x=ypx_m - y_m,octagon[id].pts[5].y=y_m;
  octagon[id].pts[6].x=x_m,octagon[id].pts[6].y=ypx_m - x_m;
  octagon[id].pts[7].x=x_m,octagon[id].pts[7].y=x_m + ymx_p;
  octagon[id].size=8;
  // if(id==0){
    // for(int i=l;i<=r;i++){printf("%f,%f\n",areas[i].x,areas[i].y);}
    // printf("-------------%d,%d\n",l,r);
  // if(id==1)for(int i=0;i<8;i++)printf("%d:%f,%f\n",i,octagon[id].pts[i].x,octagon[id].pts[i].y);
}
__global__ void initflag(int* d_in,int *octsum, int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=size||id==0)return ;
  d_in[octsum[id]-1]=1;
}
__global__ void initDouble(double*d,double DOUBLE_MAX,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=size)return;
  d[id]=DOUBLE_MAX;
}
__global__ void initPts(Pt* pts,Region*octagon,int *octsum,int *flag,int maxsize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=maxsize)return ;
  int fnum=flag[id];
  pts[id]=octagon[fnum].pts[id-octsum[fnum]];
}
__global__ void initOctagonsum(Region* octagon, int *d_in,int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>size)return ;
  if(id==size){
    d_in[id]=0;
    return ;
  }
  d_in[id]=octagon[id].size;
}
__global__ void coutoctagon(Region* octagon,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=size)return ;
  // printf("%d,%d\n",id,octagon[id].size);
  // for(int i=0;i<octagon[id].size;i++)printf("%d:%f,%f\n",i,octagon[id].pts[i].x,octagon[id].pts[i].y);
}
__global__ void convexHull(Region* octagon,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) return;
  // check pts num
  int num=octagon[id].size;
  if (num < 2) {
    return;
  }
  if (num == 2) {
    auto dist = BoundSkewTree::distance(octagon[id].pts[0], octagon[id].pts[1]);
    if (BoundSkewTree::Equal_(dist, 0)) {
      octagon->size=num-1;
    }
    return;
  }
  // calculate convex hull by Andrew algorithm
  Pt ans[16]; 
  size_t k = 0;
  for (size_t i = 0; i < octagon[id].size; ++i) {
    while (k > 1 && crossProduct(ans[k - 2], ans[k - 1], octagon[id].pts[i]) <= kEpsilon) {
      --k;
    }
    ans[k++] = octagon[id].pts[i];
  }
  for (size_t i = octagon[id].size - 1, t = k + 1; i > 0; --i) {
    while (k >= t && crossProduct(ans[k - 2], ans[k - 1], octagon[id].pts[i - 1]) <= kEpsilon) {
      --k;
    }
    ans[k++] = octagon[id].pts[i-1];
  }
  for (size_t i = 0; i < k - 1; ++i) {octagon[id].pts[i] = ans[i];}
  octagon[id].size=k-1;
  // if(id==0)
  // {
  //   printf("%d,%d\n",id, octagon[id].size);
  //   for (size_t i = 0; i <  octagon[id].size; ++i) {printf("oct:%d,%f,%f\n",id,octagon[id].pts[i].x,octagon[id].pts[i].y);}
  // }
}
__global__ void areaOnOctagonBound(Area*areas,Area*result,Region*octagon,int *bound_areas_in,int *flag,int beginID,int IDadd){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id==IDadd)bound_areas_in[IDadd]=0;
  if(id>=IDadd)return;
  int cnt=0;
  auto center = centerPt(octagon[id]);
  for(int i=areas[id+beginID].l;i<=areas[id+beginID].r;i++){
    auto arc_tan2 = std::atan2(areas[i].y - center.y, areas[i].x - center.x);
    if (arc_tan2 < 0) {
      arc_tan2 += 2 * CUDART_PI;
    }
    areas[i].val=arc_tan2;
  }
  for(int j=areas[id+beginID].l;j<=areas[id+beginID].r;j++){
    for (size_t i = 0; i < octagon[id].size; ++i) {
      Pt line[2];
      line[0]=octagon[id].pts[i];
      line[1]=octagon[id].pts[(i + 1) % octagon[id].size];
      Pt pt;
      pt.x=areas[j].x,pt.y=areas[j].y;
      // if(beginID==88&&id==3)printf("(%f,%f),lines((%f,%f),(%f,%f))\n",pt.x,pt.y,line[0].x,line[0].y,line[1].x,line[1].y);
      if (BoundSkewTree::onLine(pt, line)) {
        result[cnt+areas[id+beginID].l]=areas[j];
        // result[cnt+areas[id+beginID].l].flag=j;
        flag[cnt+areas[id+beginID].l]=id;
        cnt++;
        break;
      }
    }
  }
  // printf("%d,%f,%f\n",octagon[id].size,center.x,center.y);
  // for(int i=0;i<cnt;i++)printf("areaOnOctagonBound:%d %f,%f,%f\n",id,result[i+areas[id+beginID].l].x,result[i+areas[id+beginID].l].y,result[i+areas[id+beginID].l].val);
  bound_areas_in[id]=cnt;
}
__global__ void initboundkey(double* d_keys_in,int *d_values_in,Area*result,int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=size)return;
  d_values_in[id]=id;
  d_keys_in[id]=result[id].val;
  // printf("initboundkey:%d,%f\n",id,d_keys_in[id]);
}
__global__ void setboundArea(int* bound_areas_sum, int *bound_areas_sum_half,Area* result,Area* bound_areas,int *flag,int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=size)return;
  int segid=flag[id];
  if(segid==-1)return;
  int offset=id-bound_areas_sum[segid];
  int index1=bound_areas_sum_half[segid]+offset;
  int segment=bound_areas_sum[segid+1]-bound_areas_sum[segid];
  bound_areas[index1]=result[id];
  // printf("%d,%d,%d\n",id,index1,offset);
  if(offset<(segment/2)){
    int index2=bound_areas_sum_half[segid]+segment+offset;
    bound_areas[index2]=result[id];
  }
}
__global__ void setboundSum(int* bound_areas_in,int* bound_areas_in_1,int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id==size)bound_areas_in_1[id]=0;
  if(id>=size)return;
  bound_areas_in_1[id]=bound_areas_in[id]+bound_areas_in[id]/2;
}
__global__ void setResult(Area*areas,Area*result,Area*result1,int *flag,int *flag1,int *bound_areas_sum,int beginID,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=size)return;
  if(flag[id]==-1)return;
  // printf("flag[id]:%d %d\n",id,flag[id]);
  int l=areas[flag[id]+beginID].l;
  int index=id-l+bound_areas_sum[flag[id]];
  result[index]=result1[id];
  flag1[index]=flag[id];
  // printf("%d %d %d \n",l,id,index);
}
__global__ void setDivideArea(Area*areas,Area*bound_areas,int *bound_areas_sum_half,int offset,int *host_bound_areas,int*F,int beginID,double DOUBLE_MIN,double DOUBLE_MAX,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x; //k表示areasize，进行划分
  if(id>=size)return;
  areas[id].val=0;
  if(F[id]==-1)return;
  auto min_dist = DOUBLE_MAX;
  auto max_dist = DOUBLE_MIN;
  int k=F[id];
  int half_num=host_bound_areas[k]/2;
  if(offset>=host_bound_areas[k]){
    // f[id]=false;
    return ;
  }
  // printf("setDivideArea:%d,%d,%d\n",k,bound_areas_sum_half[k]+offset,bound_areas_sum_half[k]+offset+half_num);
  for(int i=bound_areas_sum_half[k]+offset;i<bound_areas_sum_half[k]+offset+half_num;i++){
      Pt area,ref;
      area.x=areas[id].x,area.y=areas[id].y;
      ref.x=bound_areas[i].x,ref.y=bound_areas[i].y;
      auto dist = BoundSkewTree::distance(area, ref);
      // if(area.x>16&&area.x<17)printf("(%d,%d)%f ",bound_areas_sum[k]+offset,bound_areas_sum[k]+offset+half_num-1,dist);
      min_dist = fmin(min_dist, dist);
      max_dist = fmax(max_dist, dist);
    }
  areas[id].val=max_dist + min_dist;
  // printf("\nsetDivideArea:%d,%f\n",id,areas[id].val);
}
__global__ void setSortArea(Area*areas1,Area*areas,int *d_values_out,int k){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int size=areas[k].r-areas[k].l+1;
  if(id>=size)return;
  areas[id+areas[k].l]=areas1[d_values_out[id]];
}
__global__ void setSortAreaLevel(Area*areas1,Area*areas,int *d_values_out,int *area_sum,int*F,int beginID,int pinsize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=pinsize)return;
  if(F[id]==-1)return;
  int offset=id-areas[F[id]+beginID].l;
  int index=offset+area_sum[F[id]];
  areas[id]=areas1[d_values_out[index]];
}
__global__ void setAreaOnBoundTwo(Area*areas,Region*octagon,int *ifbound,int l,int r,int left_num){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int size=r-l+1;
  if(id>=size)return;
  int side;
  if(id<left_num)side=0;
  else side=1;
  // if(side==0)printf("octagon[side].size:%d\n",octagon[side].size);
  // if(side==0)printf("%f,%f\n",areas[l+id].x,areas[l+id].y);
  // if(side==0){
  //   printf("octagon[side].size:%d\n",octagon[side].size);
  //   for (size_t i = 0; i < octagon[side].size; ++i)printf("oct:%f,%f\n",octagon[side].pts[i].x,octagon[side].pts[i].y);
  // }
  for (size_t i = 0; i < octagon[side].size; ++i) {
    Pt line[2];
    line[0]=octagon[side].pts[i];
    line[1]=octagon[side].pts[(i + 1) % octagon[side].size];
    Pt pt;
    int j=l+id;
    pt.x=areas[j].x,pt.y=areas[j].y;
    if (BoundSkewTree::onLine(pt, line)) {
      ifbound[id]=1;
      break;
    }
  }
}
__global__ void setAreaOnBoundTwolevel(Area*areas,Region*octagon,int *ifbound,int* left_num,int size,int *F,int beginID){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=size)return;
  if(F[id]==-1)return;
  int k=F[id];
  int l=areas[k+beginID].l;
  int offset=id-l;
  int side;
  if(offset<left_num[k])side=0;
  else side=1;
  // if(side==0)printf("octagon[side].size:%d\n",octagon[side].size);
  // if(side==0)printf("%f,%f\n",areas[l+id].x,areas[l+id].y);
  // if(side==0){
  //   printf("octagon[side].size:%d\n",octagon[side].size);
  //   for (size_t i = 0; i < octagon[side].size; ++i)printf("oct:%f,%f\n",octagon[side].pts[i].x,octagon[side].pts[i].y);
  // }
  // if(id==9)printf("%f,%f,octagon[side].size:%d,%d\n",areas[id].x,areas[id].y,k*2+side,octagon[k*2+side].size);
  // if(id==9)for(size_t i = 0; i < octagon[k*2+side].size; ++i)printf("%f,%f\n",octagon[k*2+side].pts[i].x,octagon[k*2+side].pts[i].y);
  for (size_t i = 0; i < octagon[k*2+side].size; ++i) {
    Pt line[2];
    line[0]=octagon[k*2+side].pts[i];
    line[1]=octagon[k*2+side].pts[(i + 1) % octagon[k*2+side].size];
    Pt pt;
    pt.x=areas[id].x,pt.y=areas[id].y;
    if (BoundSkewTree::onLine(pt, line)) {
      ifbound[id]=1;
      break;
    }
  }
}
__global__ void setNewArea(Area*areas,int* divide,int* ID,int beginID,int areasize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=areasize)return ;
  // if(areas[id+beginID].l==2005)printf("oooooooooo\n");
  // printf("%d,%d,%d,%d\n",id+beginID,divide[id],areas[id+beginID].l,areas[id+beginID].r);
  // if(divide[id]==-1)return;
  // if(id+beginID==52)printf("52:%d %d\n",areas[id+beginID].r,areas[id+beginID].l);
  // if(areas[id+beginID].r>=11&&areas[id+beginID].l<=11)printf("%d %d %d\n",id+beginID,areas[id+beginID].r,areas[id+beginID].l);
  if(areas[id+beginID].r<=areas[id+beginID].l)return;
  divide[id]=max(divide[id],1);
  divide[id]=min(divide[id],areas[id+beginID].r-areas[id+beginID].l);
  if((areas[id+beginID].r-areas[id+beginID].l+1)==2)divide[id]=1;
  int index1=beginID+areasize+ID[id];
  int index2=index1+1;
  areas[id+beginID].left=index1;
  areas[id+beginID].right=index2;
  areas[id+beginID].get_left=&areas[index1];
  areas[id+beginID].get_right=&areas[index2];
  // printf("%d,%d\n",index1,index2);
  areas[index1].l=areas[id+beginID].l;
  areas[index1].r=areas[id+beginID].l+divide[id]-1;
  areas[index2].l=areas[id+beginID].l+divide[id];
  areas[index2].r=areas[id+beginID].r;
  areas[index1].p=id+beginID;
  areas[index2].p=id+beginID;
  // if(index1==83||index2==83)printf("83 %d %d\n",areas[id+beginID].l,areas[id+beginID].r);
  // if(areas[index1].l==11)printf("%d %d %d\n",index1,areas[index1].l,areas[index1].r);
  // if(areas[index2].l==11)printf("%d %d %d\n",index2,areas[index2].l,areas[index2].r);
  // if(index1==52)printf("52:%d %d %d\n",areas[index1].l,areas[index1].r,divide[id]);
  // if(index2==52)printf("52:%d %d %d\n",areas[index2].l,areas[index2].r,divide[id]);
  if(areas[index1].r==areas[index1].l)areas[index1].cap=areas[areas[index1].l].cap,areas[index1].left=-1,areas[index1].right=-1;
  if(areas[index2].r==areas[index2].l)areas[index2].cap=areas[areas[index2].l].cap,areas[index2].left=-1,areas[index2].right=-1;
  if(areas[index1].r==areas[index1].l)areas[areas[index1].l].p=index1;
  if(areas[index2].r==areas[index2].l)areas[areas[index2].r].p=index2;
  // printf("index1 %d %d %d\n",index1,areas[index1].l,areas[index1].r);
  // printf("index2 %d %d %d\n",index2,areas[index2].l,areas[index2].r);
  // printf("%d\n",areas[11].p);
  // if(areas[index1].r==areas[index1].l&&areas[areas[index1].r].x>162&&areas[areas[index1].r].x<163)printf("%d %d\n",areas[index1].left,areas[index1].right);
  // if(areas[index2].r==areas[index2].l&&areas[areas[index2].r].x>162&&areas[areas[index2].r].x<163)printf("%d %d\n",areas[index2].left,areas[index2].right);
  // if(areas[index2].r==areas[index2].l)areas[areas[index2].r].p=id+beginID;
  // if(id+beginID==74||id+beginID==110)printf("index:%d %d,%d,%d\n",id+beginID,index1,areas[index1].l,areas[index1].r);
  // if(id+beginID==74||id+beginID==110)printf("index:%d %d,%d,%d\n",id+beginID,index2,areas[index2].l,areas[index2].r);
}
__global__ void initXY(Area*areas,double *X,double* Y,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>size)return ;
  if(id==size){
    X[id]=0,Y[id]=0;
    return ;
  }
  X[id]=areas[id].x;
  Y[id]=areas[id].y;
}
__global__ void setCenter(Area*areas,double *Xsum,double* Ysum,int begin,int end){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int size=end-begin+1;
  if(id>=size)return ;
  id+=begin;
  double x=Xsum[areas[id].r+1]-Xsum[areas[id].l]; 
  double y=Ysum[areas[id].r+1]-Ysum[areas[id].l]; 
  // if(id==104)printf("%d: %d %d %f,%f\n",id,areas[id].r,areas[id].l,x,y);
  areas[id].x=x/(areas[id].r-areas[id].l+1);
  areas[id].y=y/(areas[id].r-areas[id].l+1);
  areas[id].location=Pt(areas[id].x,areas[id].y);
  areas[id].id=id;
}
__global__ void merge(Area* areas,int beginID,int size,double DOUBLE_MAX){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int lrsize=areas[id+beginID].r-areas[id+beginID].l+1;
  if(lrsize<=1)return ;
  Area* parent=&areas[id+beginID];
  // calcJS(parent, left, right,DOUBLE_MAX);
  // parent->set_edge_len(kLeft, -1);
  // parent->set_edge_len(kRight, -1);
  // auto dist = parent->get_radius();
  // jsProcess(parent);
  // auto left_line = parent->get_line(kLeft);
  // auto right_line = parent->get_line(kRight);
  // constructMr(parent, left, right);
  // if (Geom::lineType(getJsLine(kLeft)) == LineType::kManhattan) {
  //   LOG_FATAL_IF(Geom::lineType(getJsLine(kRight)) != LineType::kManhattan) << "right js is not manhattan";
  //   if (Geom::isSegmentTrr(_ms[kLeft])) {
  //     Geom::msToLine(_ms[kLeft], left_line);
  //   }
  //   if (Geom::isSegmentTrr(_ms[kRight])) {
  //     Geom::msToLine(_ms[kRight], right_line);
  //   }
  // }
  // parent->set_line(kLeft, left_line);
  // parent->set_line(kRight, right_line);

  // parent->set_radius(dist);
  // if (parent->get_edge_len(kLeft) + parent->get_edge_len(kRight) < 0) {
  //   parent->set_cap_load(left->get_cap_load() + right->get_cap_load() + parent->get_radius() * _unit_h_cap);
  // } else {
  //   parent->set_cap_load(left->get_cap_load() + right->get_cap_load()
  //                        + (parent->get_edge_len(kLeft) + parent->get_edge_len(kRight)) * _unit_h_cap);
  // }
}
void BoundSkewTree::calcOctagonTwolevelbig(Area* areas, Region* octagon,int* left_num,double DOUBLE_MIN,double DOUBLE_MAX,int beginID,int areasize){

  for(int i=0;i<areasize;i++){
    int blockSize = 256;
    calcOctagonTwolevelbigstep<<<1, blockSize>>>(areas,octagon,left_num,DOUBLE_MIN,DOUBLE_MAX,beginID,i);
  }
}
void BoundSkewTree::calculateCap(double* capsum, double* cap){
  // 2. 确定临时存储需求
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes,
      cap, capsum, pinsize+1);
  
  // 3. 分配临时存储
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  
  // 4. 运行独占前缀和
  cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes,
      cap, capsum, pinsize+1);
  //前缀和计算cap_sum
  cudaFree(d_temp_storage);
}
void BoundSkewTree::calcOctagonnum(Region* octagon,int *octsum,int size){
  int blockSize = 256;
  int gridSize = (size+1 + blockSize - 1) / blockSize;
  int *d_in;                            // 设备端输入指针
  cudaMalloc(&d_in, (size+1) * sizeof(int));
  initOctagonsum<<<gridSize, blockSize>>>(octagon,d_in,size);
  // std::cout<<"initOctagonsum:";
  cudaDeviceSynchronize();
  // 2. 确定临时存储需求
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes,
      d_in, octsum, size+1);
  
  // 3. 分配临时存储
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  
  // 4. 运行独占前缀和
  cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes,
      d_in, octsum, size+1);
  //前缀和计算octsum
  cudaFree(d_temp_storage);
  cudaFree(d_in);
}
void BoundSkewTree::SegmentedRadixSort(Region* octagon,int *octsum, unsigned long *d_keys_in,int *d_values_in,unsigned long *d_keys_out,int *d_values_out,int num_segments,int num_items){
  // 计算临时存储需求
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes,
      d_keys_in, d_keys_out, d_values_in, d_values_out,
      num_items, num_segments, octsum, octsum + 1
  );
  // 分配临时存储
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // 执行排序
  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes,
      d_keys_in, d_keys_out, d_values_in, d_values_out,
      num_items, num_segments, octsum, octsum + 1
  );
  cudaFree(d_temp_storage);
}
void BoundSkewTree::calcflag(int *flag,int *octsum,int size){
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  int maxsize=getlastInt(octsum,size+1);
  int *d_in;                            // 设备端输入指针
  cudaMalloc(&d_in, maxsize * sizeof(int));
  cudaMemset(d_in,0,maxsize * sizeof(int));
  // std::cout<<"d_in:";
  // coutInt(d_in,maxsize);
  initflag<<<gridSize, blockSize>>>(d_in,octsum,size);
  cudaDeviceSynchronize();
    // 2. 确定临时存储需求
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_in, flag, maxsize);
    // 3. 分配临时存储
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // 4. 运行独占前缀和
  cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_in, flag, maxsize);
    //前缀和计算flag
  cudaFree(d_temp_storage);
  cudaFree(d_in);
}
// void BoundSkewTree::convexHullsort(Region* octagon,int size){
//   int blockSize = 256;
//   int gridSize = (size + blockSize - 1) / blockSize;
//   int *octsum,*flag;
//   cudaMalloc(&octsum, (size+1) * sizeof(int));
//   calcOctagonnum(octagon,octsum,size);
//   int maxsize = getlastInt(octsum,size+1);
//   // std::cout<<"flagmaxsize:"<<maxsize<<'\n';
//   cudaMalloc(&flag, maxsize * sizeof(int));
//   calcflag(flag,octsum,size);
//   // coutInt(flag,maxsize);
//   unsigned long *d_keys_in,*d_keys_out;
//   int *d_values_in,*d_values_out;
//   cudaMalloc(&d_keys_in, maxsize * sizeof(unsigned long));
//   cudaMalloc(&d_values_in, maxsize * sizeof(int));
//   cudaMalloc(&d_keys_out, maxsize * sizeof(unsigned long));
//   cudaMalloc(&d_values_out, maxsize * sizeof(int));
//   gridSize = (maxsize + blockSize - 1) / blockSize;
//   PtToKey<<<gridSize, blockSize>>>(d_keys_in,d_values_in,octagon,octsum,flag,maxsize);
//   // coutUInt(d_keys_in,maxsize);
//   cudaDeviceSynchronize();
//   SegmentedRadixSort(octagon,octsum,d_keys_in,d_values_in,d_keys_out,d_values_out,size,maxsize);
//   cudaDeviceSynchronize();
//   Pt* pts;
//   cudaMalloc(&pts, maxsize * sizeof(Pt));
//   initPts<<<gridSize, blockSize>>>(pts,octagon,octsum,flag,maxsize);
//   cudaDeviceSynchronize();
//   setconvexHull<<<gridSize, blockSize>>>(octagon,octsum, flag,d_keys_in,d_values_out,pts,maxsize);
//   cudaDeviceSynchronize();
//   cudaFree(pts);
//   cudaFree(d_keys_in);
//   cudaFree(d_keys_out);
//   cudaFree(d_values_in);
//   cudaFree(d_values_out);
//   cudaFree(octsum);
//   cudaFree(flag);
//   // std::cout<<"sort over"<<size<<std::endl;
//   // coutoctagon<<<gridSize, blockSize>>>(octagon,size);
//   // cudaDeviceSynchronize();  // 等待内核完成
//   // cudaDeviceReset();  
// }
void BoundSkewTree::convexHullsort(Region* octagon,int size){
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  octagonsort<<<gridSize, blockSize>>>(octagon,size);
}
void BoundSkewTree::Init(Pointc* pts,Pointc* clusters,int* clusters_segment,int pinnum,int clusters_num){
  net_num=clusters_num;
  cudaMalloc(&areas, (3*pinnum+10) * sizeof(Area));
  cudaMemset(areas, 0, (3*pinnum+10) * sizeof(Area));
  // std::cout<<"BoundSkewTree begin\n";
  int blockSize = 256;
  int gridSize = (pinnum + blockSize - 1) / blockSize;
  pinsize=pinnum;
  initareas<<<gridSize, blockSize>>>(areas,pts,pinsize);
  gridSize = (net_num + blockSize - 1) / blockSize;
  // coutInt(clusters_segment,clusters_num);
  initroot<<<gridSize, blockSize>>>(areas,clusters,clusters_segment,pinsize,net_num);
  clusterflag=1;
}
BoundSkewTree::BoundSkewTree(Pointc* pts,Pointc* clusters,int* clusters_segment,int pinnum,int clusters_num){
  Init(pts,clusters,clusters_segment,pinnum,clusters_num);
}
void BoundSkewTree::initAreas(){
  std::cout<<"initAreas\n";
  int size=pinsize;
  double *h_x = new double[size];
  double *h_y = new double[size];
  int cnt=0;
  for(int i=0; i<_unmerged_nodes.size(); i++) {
      h_x[i] = _unmerged_nodes[i]->x;
      h_y[i] = _unmerged_nodes[i]->y;
  }
  // 2. 分配设备内存
  double *d_x, *d_y;
  cudaMalloc(&d_x,size*sizeof(double));
  cudaMalloc(&d_y, size*sizeof(double));

  // 3. 拷贝数据到设备
  cudaMemcpy(d_x, h_x, size*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, size*sizeof(double), cudaMemcpyHostToDevice);
  // 4. 释放临时主机内存
  delete[] h_x;
  delete[] h_y;

  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  int *d_netpinnum,*netpinsum;
  cudaMalloc(&d_netpinnum,net_num*sizeof(int));
  cudaMalloc(&netpinsum,net_num*sizeof(int));
  cudaMemcpy(d_netpinnum, netpinnum, net_num*sizeof(int), cudaMemcpyHostToDevice);
  scanInt(d_netpinnum,netpinsum,net_num);
  initareas<<<gridSize, blockSize>>>(areas,d_x,d_y,size);
  gridSize = (net_num + blockSize - 1) / blockSize;

  initroot<<<gridSize, blockSize>>>(areas,d_netpinnum,netpinsum,pinsize,net_num);
  cudaDeviceSynchronize();
  cudaFree(d_x);
  cudaFree(d_y);
}
void BoundSkewTree::scanInt(int* d_in,int* Intout,int size){
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes,
      d_in, Intout, size);
  
  // 3. 分配临时存储
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  
  // 4. 运行独占前缀和
  cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes,
      d_in, Intout, size);

  cudaFree(d_temp_storage);
}
void BoundSkewTree::scanDouble(double* d_in,double* Intout,int size){
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes,
      d_in, Intout, size);
  
  // 3. 分配临时存储
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  
  // 4. 运行独占前缀和
  cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes,
      d_in, Intout, size);

  cudaFree(d_temp_storage);
}

void BoundSkewTree::boundAreaSort(Area*result,Area* bound_areas,int *bound_areas_sum,int *bound_areas_sum_half,int * bound_areas_in,int* flag,int areasize){
  int size=getlastInt(bound_areas_sum,areasize+1);
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  boundAreasort<<<gridSize, blockSize>>>(result,bound_areas_sum,areasize);
  // int* d_values_in,*d_values_out;
  // double* d_keys_in,*d_keys_out;
  // cudaMalloc(&d_values_in,size*sizeof(int));
  // cudaMalloc(&d_values_out,size*sizeof(int));
  // cudaMalloc(&d_keys_in,size*sizeof(double));
  // cudaMalloc(&d_keys_out,size*sizeof(double));
  // initboundkey<<<gridSize, blockSize>>>(d_keys_in,d_values_in,result,size);
  // cudaDeviceSynchronize();
  // // 计算临时存储需求
  // void *d_temp_storage = nullptr;
  // size_t temp_storage_bytes = 0;
  // cub::DeviceSegmentedRadixSort::SortPairs(
  //     d_temp_storage, temp_storage_bytes,
  //     d_keys_in, d_keys_out, d_values_in, d_values_out,
  //     size, areasize, bound_areas_sum, bound_areas_sum + 1
  // );
  // // 分配临时存储
  // cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // // 执行排序
  // cub::DeviceSegmentedRadixSort::SortPairs(
  //     d_temp_storage, temp_storage_bytes,
  //     d_keys_in, d_keys_out, d_values_in, d_values_out,
  //     size, areasize, bound_areas_sum, bound_areas_sum + 1
  // );
  // cudaFree(d_values_in);
  // cudaFree(d_keys_in);
  // cudaFree(d_temp_storage);

  int *bound_areas_in_1;   
  cudaMalloc(&bound_areas_in_1, (areasize+1) * sizeof(int));
  gridSize = (areasize+1 + blockSize - 1) / blockSize;
  setboundSum<<<gridSize, blockSize>>>(bound_areas_in,bound_areas_in_1,areasize);
  // coutInt(bound_areas_in,areasize);
  cudaDeviceSynchronize();
  scanInt(bound_areas_in_1,bound_areas_sum_half,areasize+1);
  cudaDeviceSynchronize();
  // coutInt(bound_areas_in,areasize+1);
  // coutInt(bound_areas_sum,areasize+1);
  // coutInt(bound_areas_sum_half,areasize+1);
  // coutInt(flag,14);
  gridSize = (size + blockSize - 1) / blockSize;
  // coutInt(flag,size);
  setboundArea<<<gridSize, blockSize>>>(bound_areas_sum,bound_areas_sum_half,result,bound_areas,flag,size);
  cudaDeviceSynchronize();
  // coutInt(bound_areas_in,areasize+1);
  int s=getlastInt(bound_areas_sum_half,areasize+1);
  // std::cout<<s<<std::endl;
  // coutAreas(bound_areas,0,s-1);
  cudaFree(bound_areas_in_1);
}
void BoundSkewTree::AreaValSort(Area*areas,int kid){
  Area*areas1;
  int size=getArearight(areas,kid)-getArealeft(areas,kid)+1;
  cudaMalloc(&areas1,size*sizeof(Area));
  Area* src = areas + getArealeft(areas,kid);
  cudaMemcpy(areas1, src, size * sizeof(Area), cudaMemcpyDeviceToDevice);
  int* d_values_in,*d_values_out;
  double* d_keys_in,*d_keys_out;
  cudaMalloc(&d_values_in,size*sizeof(int));
  cudaMalloc(&d_values_out,size*sizeof(int));
  cudaMalloc(&d_keys_in,size*sizeof(double));
  cudaMalloc(&d_keys_out,size*sizeof(double));
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  // std::cout<<"initboundkey"<<std::endl;
  initboundkey<<<gridSize, blockSize>>>(d_keys_in,d_values_in,areas1,size);
  cudaDeviceSynchronize();
  // std::cout<<"initboundkey_finish"<<std::endl;
  void  *d_temp_storage = nullptr;
  size_t  temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      d_keys_in, d_keys_out, d_values_in, d_values_out, size);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  
  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
      d_keys_in, d_keys_out, d_values_in, d_values_out, size);
  // std::cout<<"setSortArea"<<std::endl;
  gridSize = (kid + blockSize - 1) / blockSize;
  setSortArea<<<gridSize, blockSize>>>(areas1,areas,d_values_out,kid);
  cudaDeviceSynchronize();
  // std::cout<<"setSortArea_finish"<<std::endl;
  cudaFree(d_temp_storage);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(areas1);
}
void BoundSkewTree::AreaValSortLevelSmall(int* F,int beginID,int areasize){
  int blockSize = 256;
  int gridSize = (areasize + blockSize - 1) / blockSize;
  areasort<<<gridSize, blockSize>>>(areas,F,beginID,areasize);
  cudaDeviceSynchronize();
}
struct AreaValDecomposer {
    __host__ __device__ auto operator()(Val& key) const -> cuda::std::tuple<double&> {
        return {key.val};
    }
};
void BoundSkewTree::AreaValSortLevelBig(Area* areas1,int beginID,int areasize){
  Area h_areas[areasize];
  cudaMemcpy(h_areas,areas+beginID,areasize*sizeof(Area),cudaMemcpyDeviceToHost);
  Val *vals;
  cudaMalloc(&vals,pinsize*sizeof(Val));
  int blockSize = 256;
  int gridSize = (pinsize + blockSize - 1) / blockSize;
  initVals<<<gridSize, blockSize>>>(areas,vals,pinsize);
  cudaDeviceSynchronize();
  AreaValDecomposer decomposer;
  auto startsplit = std::chrono::high_resolution_clock::now();
  for(int i=0;i<areasize;i++){
    int size=h_areas[i].l-h_areas[i].r+1;
    // thrust::sort(thrust::device,areas+h_areas[i].l, areas+h_areas[i].r+1);
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, 
                                  vals + h_areas[i].l, vals + h_areas[i].l, size,decomposer);
    
    // 分配临时存储
    void* d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // 执行排序
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, 
                                  vals + h_areas[i].l, vals + h_areas[i].l, size,decomposer);
    
    cudaFree(d_temp_storage);
  }
  auto endsplit = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed4 = endsplit - startsplit;
  // std::cout<<"AreaValSortLevelBig:"<<elapsed4.count()<<'\n';
  setVals<<<gridSize, blockSize>>>(areas,areas1,vals,pinsize);
  cudaDeviceSynchronize();
  cudaMemcpy(areas,areas1,pinsize*sizeof(Area),cudaMemcpyDeviceToDevice);
  cudaFree(vals);
}
void BoundSkewTree::AreaValSortLevel(int*F,int beginID,int areasize){
  int *area_in,*area_sum;
  cudaMalloc(&area_sum,(areasize+1)*sizeof(int));
  cudaMalloc(&area_in,(areasize+1)*sizeof(int));
  int blockSize = 256;
  int gridSize = (areasize + blockSize - 1) / blockSize;
  initareasum<<<gridSize, blockSize>>>(areas,area_in,beginID,areasize);
  cudaDeviceSynchronize();
  scanInt(area_in,area_sum,areasize+1);
  // coutInt(area_in,areasize);
  int size=getlastInt(area_sum,areasize+1);
  Area*areas1;
  cudaMalloc(&areas1,size*sizeof(Area));
  gridSize = (pinsize + blockSize - 1) / blockSize;
  // coutInt(F,pinsize);
  initareasort<<<gridSize, blockSize>>>(areas1,areas,area_sum,F,beginID,pinsize);
  cudaDeviceSynchronize();
  int* d_values_in,*d_values_out;
  double* d_keys_in,*d_keys_out;
  cudaMalloc(&d_values_in,size*sizeof(int));
  cudaMalloc(&d_values_out,size*sizeof(int));
  cudaMalloc(&d_keys_in,size*sizeof(double));
  cudaMalloc(&d_keys_out,size*sizeof(double));
  gridSize = (size + blockSize - 1) / blockSize;
  // std::cout<<"initboundkey"<<std::endl;
  initboundkey<<<gridSize, blockSize>>>(d_keys_in,d_values_in,areas1,size);
  cudaDeviceSynchronize();
  // std::cout<<"initboundkey_finish"<<std::endl;
  void  *d_temp_storage = nullptr;
  size_t  temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes,
      d_keys_in, d_keys_out, d_values_in, d_values_out,
      size, areasize, area_sum, area_sum + 1
  );
  // 分配临时存储
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // 执行排序
  cub::DeviceSegmentedRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes,
      d_keys_in, d_keys_out, d_values_in, d_values_out,
      size, areasize, area_sum, area_sum + 1
  );
  // std::cout<<"setSortArea"<<std::endl;
  gridSize = (pinsize + blockSize - 1) / blockSize;
  setSortAreaLevel<<<gridSize, blockSize>>>(areas1,areas,d_values_out,area_sum,F,beginID,pinsize);
  cudaDeviceSynchronize();
  // std::cout<<"setSortArea_finish"<<std::endl;
  cudaFree(d_temp_storage);
  cudaFree(areas1);
  cudaFree(d_values_in);
  cudaFree(d_values_out);
  cudaFree(d_keys_in);
  cudaFree(d_keys_out);
  cudaFree(area_in);
  cudaFree(area_sum);
}
void BoundSkewTree::areaBound(Area* bound_areas,int *bound_areas_in,int *bound_areas_sum,int *bound_areas_sum_half,Region* octagon,int beginID,int areasize){
  // auto cap_sum = capsum[id+areas[id].size]-capsum[id-1];
  // auto half_cap = 1.0 * cap_sum / 2;
  int blockSize = 256;
  int gridSize = (areasize+1 + blockSize - 1) / blockSize;
  int resultsize=pinsize;
  int *flag,*flag1;
  Area* result,*result1;
  cudaMalloc(&flag,resultsize*sizeof(int));
  cudaMalloc(&flag1,resultsize*sizeof(int));
  cudaMemset(flag, -1, resultsize * sizeof(int));
  cudaMemset(flag1, -1, resultsize * sizeof(int));
  cudaMalloc(&result1, resultsize * sizeof(Area));
  // cudaMalloc(&flag, size* sizeof(Area));
  // std::cout<<"areaOnOctagonBound:"<<resultsize<<std::endl;
  // std::cout<<"areaOnOctagonBound1:"<<areasize<<'\n'<<"bound_areas_in_"<<'\n';
  areaOnOctagonBound<<<gridSize, blockSize>>>(areas,result1,octagon,bound_areas_in,flag,beginID,areasize);
  cudaDeviceSynchronize();
  // coutInt(bound_areas_in,areasize);
  // coutInt(flag,resultsize);
  // cudaDeviceSynchronize();
  scanInt(bound_areas_in,bound_areas_sum,areasize+1);
  int size=getlastInt(bound_areas_sum,areasize+1);
  cudaMalloc(&result, size * sizeof(Area));
  // coutInt(bound_areas_in,areasize);
  // coutInt(flag,resultsize);

  gridSize = (resultsize + blockSize - 1) / blockSize;
  setResult<<<gridSize, blockSize>>>(areas,result,result1,flag,flag1,bound_areas_sum,beginID,resultsize);  
  cudaDeviceSynchronize();
  // cudaDeviceSynchronize();
  cudaFree(result1);
  // coutInt(flag1,size);
  // coutAreas(result,0,size-1);
  boundAreaSort(result,bound_areas,bound_areas_sum,bound_areas_sum_half,bound_areas_in,flag1,areasize);//bug
  // cudaDeviceSynchronize();
  // std::cout<<"boundAreaSort"<<std::endl;
  // coutInt(bound_areas_in,areasize+1);
  cudaFree(result);
  cudaFree(flag1);
  cudaFree(flag);
}
__global__ void findMaxDistanceKernel(Area* areas, int* ifbound,int kid,int left_num,int side, int count, double* max_dist) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>0)return ;
  int l,ll;
  if(side==0)l=areas[kid].l,ll=0;
  else l=areas[kid].l+left_num,ll=left_num;
  double md=0;
  for(int i=0;i<count;i++){
    for(int j=i+1;j<count;j++){
      if (ifbound[i+ll] && ifbound[j+ll]) {
        Pt a,b;
        a.x=areas[i+l].x,a.y=areas[i+l].y;
        b.x=areas[j+l].x,b.y=areas[j+l].y;
        md=fmax(BoundSkewTree::distance(a, b),md);
      }
    }
  }
  max_dist[0]=md;
}

__global__ void filterSelectedNodes(int* ifbound, int l,int r, int* selectedIndices, int* selectedCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > r||idx<l) return;
    
    if (ifbound[idx] == 1) {
        int pos = atomicAdd(selectedCount, 1);  // 原子累加，避免竞争
        selectedIndices[pos] = idx;            // 记录选中节点的索引
    }
}
__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long* addr_as_ull = (unsigned long long*)address;
    unsigned long long old = *addr_as_ull;
    unsigned long long assumed;
    
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, 
                       (unsigned long long)__double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}
__global__ void computeMaxDistance(Area*areas, int* selectedIndices, int k,int l, double* maxDist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= k || j >= k || i >= j) return;  // 避免重复计算 (i,j) 和 (j,i)
    Pt a,b;
    a.x=areas[selectedIndices[i]+l].x;
    a.y=areas[selectedIndices[i]+l].y;
    b.x=areas[selectedIndices[j]+l].x;
    b.y=areas[selectedIndices[j]+l].y;
    double dist=BoundSkewTree::distance(a,b);

    atomicMaxDouble(maxDist, dist);  // 原子操作更新最大值
}
__global__ void Areasplitlevelbigstep(Area* areas,int *left_num,double* cap,int beginID,int areasize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=areasize)return ;
  double half_cap = (cap[areas[beginID+id].r+1]-cap[areas[beginID+id].l]) / 2.0;
    
    // 使用二分查找找到最接近 half_cap 的位置
    int l=areas[beginID+id].l;
    int r=areas[beginID+id].r;
    int low = areas[beginID+id].l;
    int high = areas[beginID+id].r;
    int best_idx = areas[beginID+id].l;
    double best_diff = fabs(cap[l] - half_cap);
    
    while (low <= high) {
        int mid = low + (high - low) / 2;
        double current_sum = cap[mid] - (l > 0 ? cap[l-1] : 0);
        double current_diff = fabs(current_sum - half_cap);
        
        if (current_diff < best_diff) {
            best_diff = current_diff;
            best_idx = mid;
        }
        
        if (current_sum < half_cap) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    
    // 检查相邻位置以获得更精确的结果
    for (int offset = -2; offset <= 2; offset++) {
        int check_idx = best_idx + offset;
        if (check_idx >= l && check_idx <= r) {
            double check_sum = cap[check_idx] - (l > 0 ? cap[l-1] : 0);
            double check_diff = fabs(check_sum - half_cap);
            if (check_diff < best_diff) {
                best_diff = check_diff;
                best_idx = check_idx;
            }
        }
    }
    
    left_num[id] = best_idx - l;  // left 是数量，不是索引
}
__global__ void Areasplitlevel(Area* areas,int *left_num,int beginID,int areasize,int * areasid,double DOUBLE_MAX){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=areasize)return ;
  // std::cout<<"begin"<<'\n';
  int l=areas[id+beginID].l;
  int r=areas[id+beginID].r;
  double allcap=0;
  for(int i=l;i<=r;i++)allcap+=areas[i].cap,areasid[i]=id;
  double half_cap=allcap/2.;
  int left = 0;
    double cap_count = 0;
    double diff = DOUBLE_MAX;
    for (size_t j = l; j < r; ++j) {
      cap_count += areas[j].cap;
      auto cur_diff = fabs(cap_count - half_cap);
      if (cur_diff < diff) {
        diff = cur_diff;
        left = j + 1-l;
      }
    }
  left_num[id]=left;
}

__global__ void findMaxDis(Area*areas,int*ifbound,int* left_num,double*max_dist,int beginID,int areasize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("%d,%d\n",id,size);
  if(id>=areasize)return ;
  int l=areas[id/2+beginID].l;
  int r=areas[id/2+beginID].r;
  int ll,rr,side;
  if((id&1)==0)ll=l,rr=l+left_num[id/2]-1,side=0;
  else ll=l+left_num[id/2],rr=r,side=1;
  int cnt=0;
  int bound[50];
  for(int i=ll;i<=rr;i++){
    if(ifbound[i]==1){
      bound[cnt]=i;
      cnt++;
    }
  }
  // if(beginID==35&&id/2==7)printf("findMaxDis:%d\n",cnt);
  // if(beginID==35&&id/2==7)printf("%d,%d\n",ll,rr);
  for(int i=0;i<cnt;i++){
    for(int j=i+1;j<cnt;j++){
      Pt a,b;
      a.x=areas[bound[i]].x,a.y=areas[bound[i]].y;
      b.x=areas[bound[j]].x,b.y=areas[bound[j]].y;
      max_dist[id]=fmax(BoundSkewTree::distance(a,b),max_dist[id]);
      // printf("%d,%f\n",id,max_dist[id]);
    }
  }
}
void BoundSkewTree::Areasplitlevelbig(Area* areas,int *left_num,int beginID,int areasize,int * areasid,double DOUBLE_MAX){
  double *cap;
  cudaMalloc(&cap,(pinsize+1)*sizeof(double));
  cudaMemset(cap, 0, (pinsize+1)*sizeof(double));
  thrust::transform(thrust::device, areas, areas + pinsize, cap, [] __device__ (const Area& p) { return p.cap; });
  thrust::exclusive_scan(thrust::device,cap, cap + pinsize+1, cap);
  // coutDouble(cap,pinsize);
  int blockSize = 256;
  int gridSize = (areasize + blockSize - 1) / blockSize;
  Areasplitlevelbigstep<<<gridSize, blockSize>>>(areas,left_num,cap,beginID,areasize);
  // coutInt(left_num,areasize);
  Area h_areas[areasize];
  cudaMemcpy(h_areas,areas+beginID,areasize*sizeof(Area),cudaMemcpyDeviceToHost);
  for(int i=0;i<areasize;i++)thrust::fill(thrust::device,areasid+h_areas[i].l, areasid + h_areas[i].r+1, i); 
  // coutInt(areasid,pinsize);
}
void BoundSkewTree::bound_diameter_level(int* left_num,int beginID,int areasize,double *cost,int *F,int level){
  Region* two;
  Pt* pts;
  cudaMalloc(&two,2*areasize*sizeof(Region));
  cudaMemset(two, 0, 2*areasize*sizeof(Region));
  cudaMalloc(&pts, 16*2*areasize* sizeof(Pt));
  cudaMemset(pts, 0, 16*2*areasize* sizeof(Pt));
  int blockSize = 256;
  int gridSize = (2*areasize + blockSize - 1) / blockSize;
  initOctagon<<<gridSize, blockSize>>>(two,pts,2*areasize);
  cudaDeviceSynchronize();

  double DOUBLE_MIN=std::numeric_limits<double>::min();
  double DOUBLE_MAX=std::numeric_limits<double>::max();
  // int l=getArealeft(areas,kid);
  // int r=getArearight(areas,kid);
  // std::cout<<"calcOctagonTwo begin:"<<kid<<' '<<l<<' '<<r<<'\n';
  // std::cout<<"calcOctagonTwolevel"<<'\n';
  // coutInt(F,pinsize);
  // coutInt(left_num,areasize);
  double DOUBLE_LOW=std::numeric_limits<double>::lowest();
  auto start = std::chrono::high_resolution_clock::now();
  if(big&&level<2){
    // std::cout<<"calcOctagonTwolevelbig\n";
    calcOctagonTwolevelbig(areas, two,left_num,DOUBLE_MIN,DOUBLE_MAX,beginID,2*areasize);
  }
  else calcOctagonTwolevel<<<gridSize, blockSize>>>(areas, two,left_num,DOUBLE_MIN,DOUBLE_MAX,beginID,2*areasize);

  // std::cout<<"calcOctagonTwolevel over:"<<' '<<l<<' '<<r<<'\n';
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  // if(big)std::cout<<"calcOctagonTwolevel:"<<elapsed.count()<<'\n';
  // std::cout<<"calcOctagonTwolevel over:"<<'\n';
  // coutInt(F,pinsize);
  // cudaDeviceReset();
  convexHullsort(two,2*areasize);
  // l=getArealeft(areas,kid);
  // r=getArearight(areas,kid);
  // std::cout<<"convexHullsortTwo over:"<<l<<' '<<r<<'\n';
  gridSize = (2*areasize + blockSize - 1) / blockSize;
  convexHull<<<gridSize, blockSize>>>(two,2*areasize);//check
  cudaDeviceSynchronize();
  int *ifbound,*bound_sum;
  cudaMalloc(&ifbound,(pinsize+1)*sizeof(int));
  cudaMalloc(&bound_sum,(areasize+1)*sizeof(int));
  cudaMemset(ifbound, 0, (pinsize+1) * sizeof(int));
  gridSize = (pinsize + blockSize - 1) / blockSize;
  // coutInt(F,pinsize);
  setAreaOnBoundTwolevel<<<gridSize, blockSize>>>(areas,two,ifbound,left_num,pinsize,F,beginID);
  cudaDeviceSynchronize();
  cudaFree(pts);
  cudaFree(two);
  double*max_dist;
  // cudaMalloc(&bound,boundsize*sizeof(int));
  cudaMalloc(&max_dist,2*areasize*sizeof(double));
  initDouble<<<gridSize, blockSize>>>(max_dist,0.,2*areasize);
  // cudaDeviceSynchronize();
  gridSize = (2*areasize + blockSize - 1) / blockSize;

  findMaxDis<<<gridSize, blockSize>>>(areas,ifbound,left_num,max_dist,beginID,2*areasize);
  cudaDeviceSynchronize();
  cudaFree(ifbound);
  gridSize = (areasize + blockSize - 1) / blockSize;
  // std::cout<<"cost:";
  costcal<<<gridSize, blockSize>>>(areas,cost,max_dist,beginID,areasize,DOUBLE_MAX);
  cudaDeviceSynchronize();
  // coutDouble(cost,areasize);
  cudaFree(max_dist);
}
double BoundSkewTree::getlastDouble(double* item,int size){
    double last_element;
    // 仅复制最后一个元素（偏移量 size-1）
    cudaMemcpy(&last_element, &item[size-1], sizeof(double), cudaMemcpyDeviceToHost);
    return last_element;
}
// int BoundSkewTree::Areasplit(Area* areas,int l,int r,int size){
//   double *capsum,*cap;
//   double *h_cap=new double[size];
//   cudaMalloc(&capsum, (size+1) * sizeof(double));
//   cudaMalloc(&cap, (size+1) * sizeof(double));  
//   int blockSize = 256;
//   int gridSize = (size + blockSize - 1) / blockSize;
//   readcap<<<gridSize, blockSize>>>(areas,l,r,cap,size+1);
//   scanDouble(cap,capsum,size+1);
//   cudaMemcpy(h_cap, cap, size*sizeof(double), cudaMemcpyDeviceToHost);
//   // std::cout<<"begin"<<'\n';
//   double half_cap=getlastDouble(capsum,size+1)/2.;
//   int left_num = 0;
//     double cap_count = 0;
//     double diff = std::numeric_limits<double>::max();
//     for (size_t j = 0; j < size - 1; ++j) {
//       cap_count += h_cap[j];
//       auto cur_diff = std::abs(cap_count - half_cap);
//       if (cur_diff < diff) {
//         diff = cur_diff;
//         left_num = j + 1;
//       }
//     }
//   return left_num;
// }
void BoundSkewTree::octagonDivide(int *ID,Area* bound_areas,int* bound_areas_in,int* bound_areas_sum,int* bound_areas_sum_half,int beginID,int areasize,int IDadd,int *F,int* areasid,int sum_size,int level){ //areasize里面的分类
  Area* areas1;
  // auto startDivide = std::chrono::high_resolution_clock::now();
  cudaMalloc(&areas1, pinsize*sizeof(Area));
  double DOUBLE_MIN=std::numeric_limits<double>::min();
  double DOUBLE_MAX=std::numeric_limits<double>::max();
  int* host_bound_areas;
  cudaMalloc(&host_bound_areas,areasize*sizeof(int)); 
  cudaMemcpy(host_bound_areas, bound_areas_in, areasize * sizeof(int), cudaMemcpyDeviceToDevice);
  // coutInt(bound_areas_in,areasize);
  int* host_bound_areas1 = new int[areasize];
  cudaMemcpy(host_bound_areas1, bound_areas_in, areasize * sizeof(int), cudaMemcpyDeviceToHost);
  // double*min_cost;
  // cudaMalloc(&min_cost,areasize*sizeof(double)); 
  // cudaMemset(min_cost,DOUBLE_MAX,areasize*sizeof(double));
  int *divide;
  double*min_cost;
  cudaMalloc(&min_cost,areasize*sizeof(double)); 
  cudaMalloc(&divide,areasize*sizeof(int)); 
  // cudaMemset(min_cost,DOUBLE_MAX,areasize*sizeof(double));
  int blockSize = 256;
  int gridSize = (areasize + blockSize - 1) / blockSize;
  initDouble<<<gridSize, blockSize>>>(min_cost,DOUBLE_MAX,areasize);
  cudaDeviceSynchronize();
  cudaMemset(divide,-1,areasize*sizeof(int));
  int maxIter=0;
  for(int i=0;i<areasize;i++)maxIter=max(maxIter,host_bound_areas1[i]);
  // std::cout<<"maxIter:"<<maxIter<<'\n';


  // auto endDivide = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsedDivide = endDivide - startDivide;
  // std::cout<<"DivideBegin_level_time:"<<elapsedDivide.count()<<'\n';
  double time=0,time1=0,time2=0,time3=0,time4=0;
  int *costup;
  cudaMalloc(&costup,areasize*sizeof(int));
  // coutDouble(min_cost,areasize);
  // coutAreas(bound_areas,0,host_bound_areas1[0]-1);
  // if(big){
  //   if(maxIter>10)for(int i=0;i<areasize;i++)std::cout<<"maxIter:"<<host_bound_areas1[i]<<'\n';
  // }
  Area* areas2;
  cudaMalloc(&areas2,pinsize*sizeof(Area));
  // std::cout<<sum_size<<' '<<areasize<<'\n';
  for(int i=0;i<maxIter;i++){
    // auto ref_set = std::vector<Area*>(bound_areas.begin() + i, bound_areas.begin() + i + half_num);
      gridSize = (pinsize + blockSize - 1) / blockSize;
      auto start1 = std::chrono::high_resolution_clock::now();
      setDivideArea<<<gridSize, blockSize>>>(areas,bound_areas,bound_areas_sum_half,i,host_bound_areas,F,beginID,DOUBLE_MIN,DOUBLE_MAX,pinsize);
      cudaDeviceSynchronize();
      auto end1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed1 = end1 - start1;
      time1+=elapsed1.count();
      cudaDeviceSynchronize();
      auto startsort = std::chrono::high_resolution_clock::now();

      // if(big&&level<0)AreaValSortLevelBig(areas2,beginID,areasize);
      // else if(level<2||(big&&level<9))AreaValSortLevel(F,beginID,areasize);  //4380
      // else AreaValSortLevelSmall(F,beginID,areasize);
      // std::cout<<sum_size<<' '<<areasize<<'\n';
      // if(areasize<16)AreaValSortLevelBig(areas2,beginID,areasize);
      if((sum_size/areasize)>64)AreaValSortLevel(F,beginID,areasize);  
      else AreaValSortLevelSmall(F,beginID,areasize);
      
      auto endsort = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed3 = endsort - startsort;
      time3+=elapsed3.count();
      // coutAreas(areas,0,16);
      // std::quick_exit(EXIT_FAILURE); 
      cudaDeviceSynchronize();
      // coutAreas(areas,0,3);  //ok
      auto startsplit = std::chrono::high_resolution_clock::now();
      int* left_num;
      cudaMalloc(&left_num,areasize*sizeof(int));
      gridSize = (areasize + blockSize - 1) / blockSize;
      cudaMemset(areasid, 0, pinsize*sizeof(int));
      if(big&&level<5){
        Areasplitlevelbig(areas,left_num,beginID,areasize,areasid,DOUBLE_MAX);
      }
      else Areasplitlevel<<<gridSize, blockSize>>>(areas,left_num,beginID,areasize,areasid,DOUBLE_MAX); //0.0001-0.1
      cudaDeviceSynchronize();
      // if(big)coutInt(left_num,areasize);
      auto endsplit = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed4 = endsplit - startsplit;
      time4+=elapsed4.count();

      auto start = std::chrono::high_resolution_clock::now();
      double *cost;
      cudaMalloc(&cost,areasize*sizeof(double));
      // std::cout<<"bound_diameter_level begin\n";
      bound_diameter_level(left_num,beginID,areasize,cost,F,level);//0.06 -0.1
      cudaDeviceSynchronize();
      // coutDouble(cost,areasize);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      time+=elapsed.count();
      // std::cout<<"bound_diameter over"<<'\n';
      gridSize = (areasize + blockSize - 1) / blockSize;

      start = std::chrono::high_resolution_clock::now();
      
      if(true){
        costUpdateBig<<<gridSize, blockSize>>>(areas,areas1,divide,cost,min_cost,i,host_bound_areas,left_num,beginID,costup,areasize);
        // coutInt(costup,areasize);
        // coutInt(areasid,pinsize);
        cudaDeviceSynchronize();
        gridSize = (pinsize + blockSize - 1) / blockSize;
        costUpdateBigareas<<<gridSize, blockSize>>>(areas,areas1,areasid,costup,beginID,pinsize);
      }
      else costUpdate<<<gridSize, blockSize>>>(areas,areas1,divide,cost,min_cost,i,host_bound_areas,left_num,beginID,areasize);
      // std::cout<<"area1:";
      // coutAreas(areas1,0,3);//check
      cudaDeviceSynchronize();
      end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed2 = end - start;
      time2+=elapsed2.count();
      // for(int i=0;i<areasize;i++)std::cout<<divide[i]<<' ';
      // coutAreas(areas,l,r);
      // printf("curr_min_cost:%d,%f\n",size,min_cost);
      cudaFree(cost);
      cudaFree(left_num);
  }

  // if(big){
  // std::cout<<"setDivideArea:"<<time1<<'\n';
  // std::cout<<"bound_diameter_level:"<<time<<'\n';
  // std::cout<<"AreaValSort_time:"<<time3<<'\n';
  // std::cout<<"Areasplitlevel:"<<time4<<'\n';
  // std::cout<<"costUpdate_time:"<<time2<<'\n';
  // }
  // cudaDeviceSynchronize();
  auto startafter = std::chrono::high_resolution_clock::now();
  gridSize = (pinsize + blockSize - 1) / blockSize;
  // coutInt(F,pinsize);
  // coutDouble(min_cost,areasize);
  copylevel<<<gridSize, blockSize>>>(areas,areas1,F,beginID,pinsize);
  cudaDeviceSynchronize();

  cudaFree(host_bound_areas);

  gridSize = (areasize + blockSize - 1) / blockSize;
  // coutInt(divide,areasize);
  setNewArea<<<gridSize, blockSize>>>(areas,divide,ID,beginID,areasize);
  cudaDeviceSynchronize();
  // if(beginID>1000)cudaDeviceReset();
  cudaFree(divide);
  cudaFree(areas1);
  cudaFree(min_cost);
  cudaFree(costup);
  auto endafter = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsedafter = endafter - startafter;
  // if(big)std::cout<<"Divideafter_time:"<<elapsedafter.count()<<'\n';
  // coutAreaslr(areas,18,19);
}
void BoundSkewTree::calculateCenter(int begin,int end){//闭区间
  // std::cout<<"calculateCenter"<<std::endl;
  // std::cout<<begin<<' '<<end<<std::endl;
  double *X,*Y,*Xsum,*Ysum;
  int size=end-begin+1;
  cudaMalloc(&X,(pinsize+1)*sizeof(double));
  cudaMalloc(&Y,(pinsize+1)*sizeof(double));
  cudaMalloc(&Xsum,(pinsize+1)*sizeof(double));
  cudaMalloc(&Ysum,(pinsize+1)*sizeof(double));
  int blockSize=256;
  int gridSize = ((pinsize+1) + blockSize - 1) / blockSize;
  initXY<<<gridSize, blockSize>>>(areas,X,Y,pinsize);
  cudaDeviceSynchronize();
  // coutDouble(X,pinsize+1);
  scanDouble(X,Xsum,pinsize+1);
  scanDouble(Y,Ysum,pinsize+1);
  // coutDouble(Xsum,pinsize+1);
  // Pt* pts;
  // cudaMalloc(&pts, 16*size * sizeof(Pt));
  // initconvex<<<gridSize, blockSize>>>(areas,pts,size);
  gridSize = ((size+1) + blockSize - 1) / blockSize;
  setCenter<<<gridSize, blockSize>>>(areas,Xsum,Ysum,begin,end);
  cudaDeviceSynchronize();
  cudaFree(X);
  cudaFree(Y);
  cudaFree(Xsum);
  cudaFree(Ysum);
}
void BoundSkewTree::biPartition(Region* octagon,Pt* pts,int *ID,Area* bound_areas,int *bound_areas_in,int *bound_areas_sum,int *bound_areas_sum_half){
  int blockSize = 256;
  int size=pinsize;
  int gridSize = (size + blockSize - 1) / blockSize;
  double DOUBLE_MIN=std::numeric_limits<double>::min();
  double DOUBLE_MAX=std::numeric_limits<double>::max();
  int *F;
  cudaMalloc(&F,size*sizeof(int));
  if(clusterflag==0)initAreas();
  cudaDeviceSynchronize();
  gridSize = (size+1 + blockSize - 1) / blockSize;
  initOctagon<<<gridSize, blockSize>>>(octagon,pts,size+1);
  cudaDeviceSynchronize();
  levelsum = new int[1000];
  int level=0,beginID=pinsize,beforeIDsize=net_num;
  // levelsum[0]=net_num;
  levelsum[0]=beginID;
  int* areasid,*avg_size;
  cudaMalloc(&areasid,size*sizeof(int));
  cudaMalloc(&avg_size,pinsize*sizeof(int));
  // std::cout<<levelsum[0]<<'\n';
  while(true){
    // cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    int IDadd=calculateID(ID,avg_size,beforeIDsize,size,beginID);
    int sum_size = thrust::reduce(thrust::device,avg_size, avg_size + beforeIDsize, 0, thrust::plus<int>());
    // std::cout<<sum<<'\n';
    // std::cout<<"IDadd:"<<IDadd<<' '<<beforeIDsize<<std::endl;
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end2 - start;
    // std::cout << "calculateID-level execution time: " << elapsed2.count() << " seconds" << std::endl;
    if(IDadd==0)break;
    // coutDouble(capsum,pinsize);
    cudaMemset(F,-1,size*sizeof(int));
    gridSize = (beforeIDsize + blockSize - 1) / blockSize;
    double DOUBLE_LOW=std::numeric_limits<double>::lowest();
    double DOUBLE_MIN=std::numeric_limits<double>::min();
    calcOctagon<<<gridSize, blockSize>>>(areas, octagon,beginID,beforeIDsize,DOUBLE_MIN,DOUBLE_MAX,F);   //从0开始编号
    cudaDeviceSynchronize();
    convexHullsort(octagon,beforeIDsize);
    cudaDeviceSynchronize();
    convexHull<<<gridSize, blockSize>>>(octagon,beforeIDsize);
    cudaDeviceSynchronize();
    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed4 = end3 - end2;
    // if(big)std::cout << "convexHull level execution time: " << elapsed4.count() << " seconds" << std::endl;
    // cudaDeviceReset();  // 确保输出显示
    // std::cout<<"areaBound"<<std::endl;
    //sum是加上一半后的
    areaBound(bound_areas,bound_areas_in,bound_areas_sum,bound_areas_sum_half,octagon,beginID,beforeIDsize);
    cudaDeviceSynchronize();
    // std::cout<<"areaBound over"<<std::endl;
    auto end1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed1 = end1 - end3;
    // std::cout << "areaBound level execution time: " << elapsed1.count() << " seconds" << std::endl;
    octagonDivide(ID,bound_areas,bound_areas_in,bound_areas_sum,bound_areas_sum_half,beginID,beforeIDsize,IDadd,F,areasid,sum_size,level);
    // std::cout<<"octagonDivide over"<<std::endl;
    // if(beginID==88)break;
    level++;
    beginID+=beforeIDsize;
    beforeIDsize=IDadd;
    levelsum[level]=beginID;
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - end1;
    // std::cout<<levelsum[level]<<'\n';
    // std::cout << "octagonDivide onelevel execution time: " << elapsed.count() << " seconds" << std::endl;
  }
  cudaFree(F);
  cudaFree(avg_size);
  cudaFree(areasid);
  maxlevel=level;
  maxlevel++;
  levelsum[maxlevel]=levelsum[level]+beforeIDsize;

  calculateCenter(size,levelsum[maxlevel]-1);
  // coutAreas(areas,0,3);
}
// void BoundSkewTree::recursiveBottomUp(int *levelsum){
//   double DOUBLE_MAX=std::numeric_limits<double>::max();
//   for(int i=maxlevel;i>0;i--){
//     int blockSize = 256;
//     int size=levelsum[i]-levelsum[i-1];
//     int gridSize = (size + blockSize - 1) / blockSize;
//     merge<<<gridSize, blockSize>>>(areas,levelsum[i-1],0,size,DOUBLE_MAX);
//   }
// }
// void BoundSkewTree::embedding(int* levelsum){
//   for(int i=0;i<maxlevel;i++){
//     int blockSize = 256;
//     int size=levelsum[i+1]-levelsum[i];
//     int gridSize = (size + blockSize - 1) / blockSize;
//     embeddingChild<<<gridSize, blockSize>>>(areas,levelsum[i],0,size);
//     embeddingChild<<<gridSize, blockSize>>>(areas,levelsum[i],1,size);
//   }
//   for(int i=maxlevel;i>0;i--){
//     int blockSize = 256;
//     int size=levelsum[i]-levelsum[i-1];
//     int gridSize = (size + blockSize - 1) / blockSize;
//     updateTiming<<<gridSize, blockSize>>>(areas,levelsum[i-1],0,size);
//   }
// }
void BoundSkewTree::coutDouble(double* d,int size){
  double *h = new double[size];
  cudaMemcpy(h, d, (size) * sizeof(double), cudaMemcpyDeviceToHost);
  std::cout << "values: ";
  for (int i = 0; i < size ; i++) {
      std::cout << h[i] << " ";
  }
  std::cout << std::endl;
}
void BoundSkewTree::coutInt(int* d,int size){
  int *h = new int[size];
  cudaMemcpy(h, d, (size) * sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "values: ";
  for (int i = 0; i < size ; i++) {
    if(h[i]>100000)std::quick_exit(EXIT_FAILURE); 
      std::cout << h[i] << " ";
  }
  std::cout << std::endl;
}
void BoundSkewTree::coutUInt(unsigned long* d,int size){
  unsigned long *h = new unsigned long[size];
  cudaMemcpy(h, d, (size) * sizeof(unsigned long), cudaMemcpyDeviceToHost);
  std::cout << "key: ";
  for (int i = 0; i < size ; i++) {
    // if(h[i]>100000)std::quick_exit(EXIT_FAILURE); 
      std::cout << h[i] << " ";
  }
  std::cout << std::endl;
}
void BoundSkewTree::coutAreaslr(Area* d_a, int l, int r) {
  // 1. 参数验证
  if (l > r || l < 0 || d_a == nullptr) {
      std::cerr << "Invalid range or null pointer" << std::endl;
      return;
  }

  // 2. 计算元素数量
  const size_t count = r - l + 1;
  const size_t bytes = count * sizeof(Area);

  // 3. 分配主机内存
  Area* h_a = new Area[count];

  // 4. 执行内存拷贝
  cudaError_t err = cudaMemcpy(h_a, d_a + l, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) 
               << " (code " << err << ")" << std::endl;
      delete[] h_a;
      return;
  }
  int f[17000];
  for(int i=0;i<17000;i++)f[i]=0;
  // 5. 打印结果
  for (size_t i = 0; i < count; ++i) {
      // std::cout << h_a[i].l << " " << h_a[i].r << "\n";
      for(int j=h_a[i].l;j<=h_a[i].r;j++)f[j]=1;
  }
  for(int i=0;i<17000;i++)if(f[i]==0)std::cout<<i<<' ';
  // 6. 释放内存
  delete[] h_a;
}
void BoundSkewTree::coutAreassum(Area* d_a, int l, int r) {
  // 1. 参数验证
  if (l > r || l < 0 || d_a == nullptr) {
      std::cerr << "Invalid range or null pointer" << std::endl;
      return;
  }

  // 2. 计算元素数量
  const size_t count = r - l + 1;
  const size_t bytes = count * sizeof(Area);

  // 3. 分配主机内存
  Area* h_a = new Area[count];

  // 4. 执行内存拷贝
  cudaError_t err = cudaMemcpy(h_a, d_a + l, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) 
               << " (code " << err << ")" << std::endl;
      delete[] h_a;
      return;
  }

  // 5. 打印结果
  
  int cnt=0;
  for (size_t i = 0; i < count; ++i) {
       int s=h_a[i].r-h_a[i].l+1;
       cnt+=s;
  }
  // 6. 释放内存
  delete[] h_a;
  std::cout<<"l-r_sum:"<<l<<' '<<r<<':'<<cnt<<'\n';
}
void BoundSkewTree::coutAreas(Area* d_a, int l, int r) {
  std::cout<<l<<'-'<<r<<'\n';
  // 1. 参数验证
  if (l > r || l < 0 || d_a == nullptr) {
      std::cerr << "Invalid range or null pointer" << std::endl;
      return;
  }

  // 2. 计算元素数量
  const size_t count = r - l + 1;
  const size_t bytes = count * sizeof(Area);

  // 3. 分配主机内存
  Area* h_a = new Area[count];

  // 4. 执行内存拷贝
  cudaError_t err = cudaMemcpy(h_a, d_a + l, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) 
               << " (code " << err << ")" << std::endl;
      delete[] h_a;
      return;
  }

  // 5. 打印结果
  for (size_t i = 0; i < count; ++i) {
      std::cout << h_a[i].x << " " << h_a[i].y <<' '<<h_a[i].val<< "\n";
  }

  // 6. 释放内存
  delete[] h_a;
}
int BoundSkewTree::getArealeft(Area* areas,int k){
  Area host_area; 
  cudaMemcpy(&host_area, &areas[k], sizeof(Area), cudaMemcpyDeviceToHost);
  return host_area.l;
}
int BoundSkewTree::getArearight(Area* areas,int k){
  Area host_area; 
  cudaMemcpy(&host_area, &areas[k], sizeof(Area), cudaMemcpyDeviceToHost);
  return host_area.r;
}
void BoundSkewTree::bottomUp(){
  Region* octagon;
  Pt* pts;
  int *ID;
  Area* bound_areas;
  cudaMalloc(&octagon, (pinsize+1)*sizeof(Region));
  cudaMemset(octagon, 0, (pinsize+1)*sizeof(Region));

  cudaMalloc(&pts, 16*(pinsize+1) * sizeof(Pt));
  cudaMemset(pts, 0, 16*(pinsize+1) * sizeof(Pt));

  cudaMalloc(&ID, (pinsize+1)*sizeof(int));
  cudaMalloc(&bound_areas, (pinsize+pinsize/2)*sizeof(Area));
  cudaMemset(bound_areas, 0, (pinsize+pinsize/2)*sizeof(Area));
  int *bound_areas_in,*bound_areas_sum,*bound_areas_sum_half;   
  cudaMalloc(&bound_areas_in, (pinsize+1) * sizeof(int));
  cudaMalloc(&bound_areas_sum, (pinsize+1) * sizeof(int));
  cudaMalloc(&bound_areas_sum_half, (pinsize+1) * sizeof(int));

  cudaDeviceSynchronize();
  if(clusterflag==0)cudaMalloc(&areas, (3*pinsize+10) * sizeof(Area));
  // auto start = std::chrono::high_resolution_clock::now();
  biPartition(octagon,pts,ID,bound_areas,bound_areas_in,bound_areas_sum,bound_areas_sum_half);
  // recursiveBottomUp();
  cudaDeviceSynchronize();
  // auto end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed = end - start;
  // std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;
  cudaFree(pts);
  cudaFree(octagon);
  cudaFree(ID);
  cudaFree(bound_areas);
  cudaFree(bound_areas_in);
  cudaFree(bound_areas_sum);
  cudaFree(bound_areas_sum_half);
  // cudaDeviceSynchronize();
  // start = std::chrono::high_resolution_clock::now();
  // double *capsum;
  // cudaMalloc(&capsum, (10000) * sizeof(double));
  // cudaDeviceSynchronize();
  // end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed1 = end - start;
  // std::cout << "Total execution time: " << elapsed1.count()*10000 << " seconds" << std::endl;
}
void BoundSkewTree::run(){
  auto start = std::chrono::high_resolution_clock::now();
  // std::cout<<"run\n";
  bottomUp();
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  // std::cout << "bottomUp execution time: " << elapsed.count() << " seconds" << std::endl;
  // topDown(levelsum);
  // auto pins = _load_pins;
  // pins.push_back(_root_buf->get_driver_pin());
  // TreeBuilder::localPlace(pins);
}
}