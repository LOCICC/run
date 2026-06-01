#include "GPUClustering.h"
// #include "BoundSkewTree.h"
#include "TreeMaker.h"
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h> 
#include <thrust/reduce.h>
#include <thrust/transform.h>
// #include "MinCostFlow.hh"
#include <ctime>
namespace gcts {
__device__  double getUnitCap;
__device__  double getUnitRes;
__device__  double getDbUnit;
struct sumAll
{
    __device__ __forceinline__
    Pointc operator()(const Pointc &a, const Pointc &b) const {
        return Pointc(a.x+b.x,a.y+b.y,a.count+b.count,a.cap+b.cap);
    }
};  
__device__ double calcDist(const Pointc& p1, const Pointc& p2){
    double dist = 0;
    dist = fabs(p1.x - p2.x) + fabs(p1.y - p2.y);
    return dist;
}
__global__ void initfdom(int *fdom,int num){
  curandState state;
  curand_init(0, 0, 0, &state);
  for(int i=0;i<num;i++)fdom[i]=(int)32*curand_uniform(&state);
}
__global__ void initCostMat(Pointc *pts,Pointc *buffers,double* cost_matrices,int inst_num,int node_num,int max_fanout,int size,int flag,int *fdom){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // curandState state;
  // curand_init(0, id, 0, &state);
  if(id>=size)return ;
  // if(flag){
  //   // cost_matrices[id]=*MAX-cost_matrices[id];
  //   cost_matrices[id]=3000.-cost_matrices[id]+fdom[id%500];
  //   int p=id%(node_num*node_num);
  //   int ptsid=p/node_num;
  //   int clusrerid=(p%node_num)/max_fanout;
  //   // if(cost_matrices[id]<5.)printf("MAX-%d %d :%f %f\n",ptsid,clusrerid,cost_matrices[id],*MAX);
  //   return ;
  // }
  if(id>=inst_num*node_num){
    int p=id%(node_num*node_num);
    int ptsid=p/node_num;
    int clusrerid=p%node_num;
    cost_matrices[id]=(double)30*fdom[ptsid%1000]+(double)20*fdom[clusrerid%1000]+(double)fdom[p%1000];
    // printf("%d %d %d :%f\n",size,ptsid,clusrerid,cost_matrices[id]);
    return;
  }
  int ptsid=id/node_num;
  int clusrerid=(id%node_num)/max_fanout;
  cost_matrices[id]=3000.-(fabs(pts[ptsid].x-buffers[clusrerid].x)+fabs(pts[ptsid].y-buffers[clusrerid].y))+fdom[id%1000];
  // if(ptsid==4099&&(id%node_num)==4373)printf("%d %d %d :%f\n",id,ptsid,id%node_num,cost_matrices[id]);
}
__global__ void initDistances(double* distances, int n, double initValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        distances[idx] = initValue;  // 可自定义初始值
    }
}
__global__ void calculateDistances(Pointc * pts,Pointc * centers,double* distances, int center_num,int size, double DOUBLE_MAX,double Dmax,bool flag) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >=size) return;
    if (center_num==0){
      distances[id]=1.;
      return ;
    }
    if(!flag){
      double min_distance = calcDist(pts[id], centers[0]);
      for (size_t j = 1; j < center_num; j++) {
        double distance = calcDist(pts[id], centers[j]);
        min_distance = fmin(min_distance, distance);
      }
      distances[id] = min_distance * min_distance;  // square distance
    }
    else distances[id]/=Dmax;
}
__device__ int sample_discrete_distribution(double* weights, int n, curandState* state) {
    // 1. 计算总权重
    double total_weight = 0.0;
    for (int i = 0; i < n; ++i) {
        total_weight += weights[i];
    }
    // 2. 生成随机数并选择
    double r = curand_uniform(state) * total_weight;
    // printf("%f\n",r);
    double cumulative = 0.0;
    for (int i = 0; i < n; ++i) {
        cumulative += weights[i];
        if (r <= cumulative) {
            return i;
        }
    }
    return n - 1; // 防止浮点误差
}
__device__ int sample_discrete_distribution(double* weights, int n, float choose) {
    // 1. 计算总权重
    double total_weight = 0.0;
    for (int i = 0; i < n; ++i) {
        total_weight += weights[i];
    }
    // 2. 生成随机数并选择
    double r = choose * total_weight;
    double cumulative = 0.0;
    for (int i = 0; i < n; ++i) {
        cumulative += weights[i];
        if (r <= cumulative) {
            return i;
        }
    }
    return n - 1; // 防止浮点误差
}
__global__ void initBegincenter(Pointc * pts,Pointc * centers,double* distances,int center_num,int size, double DOUBLE_MAX){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >=size) return;
  __shared__ curandState state;
  if(id==0){
    curand_init(0, id, 0, &state);
    // for(int i=0;i<center_num;i++)choose[i]=curand_uniform(&state);
    // for(int i=0;i<center_num;i++)printf("%f\n",choose[i]);
  }
  __syncthreads();
  int cnt=0;
  while(cnt<center_num){
    distances[id]=1e15;
    for(int i=0;i<cnt;i++)distances[id]=fmin(calcDist(pts[id],centers[i]),distances[i]);
    __syncthreads();
    if(id==0){
      int index=sample_discrete_distribution(distances, size, &state);
      // printf("index:%d\n",index);
      centers[cnt] = pts[index];
    }
    __syncthreads();
    cnt++;
  }
}
__global__ void setClusternum(Pointc* clusters,int *clusters_segment,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >=size) return;
  // printf("setClusternum:%d\n",clusters[id].count);
  clusters_segment[id]=clusters[id].count;
}
__global__ void updateAssignments(Pointc* pts,int* assignments,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >=size) return;
  assignments[id]=pts[id].clusterid;
}
__device__ double calcVarianceK(Pointc* clusters,int k){
  double sum=0;
  for(int i=0;i<k;i++)sum+=clusters[i].cap;
  double mean = sum / k;
  double variance=0;
  for(int i=0;i<k;i++)variance+=((clusters[i].cap-mean)*(clusters[i].cap-mean));
  return variance/k;
}
__global__ void kMeansPlusSmallStep(Pointc * pts,Pointc* clusters,Pointc* best_clusters,int* clusters_segment,int* best_clusters_segment,int*k_segment,int *k,int size,
    Pointc *new_centers,Pointc *new_pts,double *dis,int *assignments,int*cluster_k_segment,int *cnt1){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >=size) return;
  best_clusters[k_segment[id]]=clusters[id];
  best_clusters_segment[k_segment[id]]=clusters_segment[id];
  if(k[id]<=1)return;
  int cnt=0;
  curandState state;
  curand_init(0, id, 0, &state);  // 用不同的序列号初始化
  while(cnt<k[id]){
    for(int i=clusters_segment[id];i<clusters_segment[id]+clusters[id].count;i++){
      dis[i]=1e15;
      for(int j=0;j<cnt;j++)dis[i]=fmin(calcDist(pts[i],new_centers[j]),dis[i]);
    }
    int index=sample_discrete_distribution(&dis[clusters_segment[id]], clusters[id].count, &state);
    new_centers[cnt+k_segment[id]] = pts[index+clusters_segment[id]];
    cnt++;
  }
  // for(int i=0;i<k[id];i++)printf("center:%f,%f\n",new_centers[i+k_segment[id]].x,new_centers[i+k_segment[id]].y);
  int num_iterations=0;
  int no_change=0;
  // int assignments[32];
  double prev_cap_variance = 1e38;
  while (num_iterations++ < 5 && no_change++ < 5) {
    // Assignment step
    for(int i=clusters_segment[id];i<clusters_segment[id]+clusters[id].count;i++){
      double min_distance = calcDist(pts[i],new_centers[k_segment[id]]);
      int min_center_index = 0;
      for(int j=1;j<k[id];j++){
        double distance = calcDist(pts[i],new_centers[j+k_segment[id]]);
        if (distance < min_distance) {min_distance = distance;min_center_index = j;}
      }
      assignments[i]=min_center_index;
    }
    for(int i=0;i<k[id];i++)new_centers[i+k_segment[id]].cap=0,new_centers[i+k_segment[id]].x=0,new_centers[i+k_segment[id]].y=0,new_centers[i+k_segment[id]].count=0;
    for(int i=clusters_segment[id];i<clusters_segment[id]+clusters[id].count;i++){
      int center_index = assignments[i];
      new_centers[center_index+k_segment[id]].cap+=pts[i].cap;
      new_centers[center_index+k_segment[id]].x+=pts[i].x;
      new_centers[center_index+k_segment[id]].y+=pts[i].y;
      new_centers[center_index+k_segment[id]].count++;
    } 
    for(int i=0;i<k[id];i++){
      if(new_centers[i+k_segment[id]].count==0){
        float rand_val = curand_uniform(&state);
        int new_index = static_cast<int>(rand_val * clusters[id].count);
        new_centers[i+k_segment[id]]=pts[new_index],new_centers[i+k_segment[id]].count=0;
      }
      else new_centers[i+k_segment[id]].x/=new_centers[i+k_segment[id]].count,new_centers[i+k_segment[id]].y/=new_centers[i+k_segment[id]].count;
    }
    // for(int i=0;i<k[id];i++)printf("new_centers:%f,%f\n",new_centers[i+k_segment[id]].x,new_centers[i+k_segment[id]].y);
    double cap_variance = calcVarianceK(&new_centers[k_segment[id]],k[id]);
    // printf("cap_variance:%f\n",cap_variance);
    // int cluster_k_segment[32],cnt1[32];
    for(int i=0;i<k[id];i++){
      cnt1[i+k_segment[id]]=0;
      if(i==0)cluster_k_segment[i+k_segment[id]]=0;
      else cluster_k_segment[i+k_segment[id]]=cluster_k_segment[i-1+k_segment[id]]+new_centers[i-1+k_segment[id]].count;
    }
    // printf("cluster_k_segment over\n");
    if (cap_variance < prev_cap_variance) {
      prev_cap_variance = cap_variance;
      for(int i=clusters_segment[id];i<clusters_segment[id]+clusters[id].count;i++){
        int center_index = assignments[i];
        // printf("new_pts:%d %d\n",i,center_index);
        int index=cluster_k_segment[center_index+k_segment[id]]+cnt1[center_index+k_segment[id]]++;
        new_pts[index+clusters_segment[id]]=pts[i];
        // printf("new_pts:%d %d\n",i,index+clusters_segment[id]);
      }
      for(int i=0;i<k[id];i++){
        // printf("%d %d \n",k_segment[id]+i,i);
        // printf("%f %f %f\n",new_centers[i].x,new_centers[i].y,new_centers[i].cap);
        best_clusters[k_segment[id]+i]=new_centers[k_segment[id]+i];
        // printf("best_clusters:%d %d %f %f %f %d\n",k_segment[id]+i,i,best_clusters[k_segment[id]+i].x,best_clusters[k_segment[id]+i].y,best_clusters[k_segment[id]+i].cap,best_clusters[k_segment[id]+i].count);
        best_clusters_segment[k_segment[id]+i]=cluster_k_segment[k_segment[id]+i];
        // printf("best_clusters:%d %d %f %f %f %d\n",k_segment[id]+i,i,best_clusters[k_segment[id]+i].x,best_clusters[k_segment[id]+i].y,best_clusters[k_segment[id]+i].cap,best_clusters[k_segment[id]+i].count);
      }
      no_change = 0;
    }
  }
  //   for(int i=clusters_segment[id];i<clusters_segment[id]+clusters[id].count;i++)printf("[%d,%d],pts_kMeansPlusSmall:%f,%f\n",clusters_segment[id],clusters_segment[id]+clusters[id].count-1,pts[i].x,pts[i].y);
  // for(int i=clusters_segment[id];i<clusters_segment[id]+clusters[id].count;i++)printf("[%d,%d],kMeansPlusSmall:%f,%f\n",clusters_segment[id],clusters_segment[id]+clusters[id].count-1,new_pts[i].x,new_pts[i].y);
  for(int i=clusters_segment[id];i<clusters_segment[id]+clusters[id].count;i++){pts[i]=new_pts[i];}
}
__global__ void setAssignmentsMax(Pointc * pts,Pointc * buffers ,int* buffers_segment,int max_out, int center_num,int size,double DOUBLE_MAX) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >=size) return;
    // printf("setAssignmentsMax:%d,%d\n",pts[id].clusterid,buffers[pts[id].clusterid].count);
    if(buffers[pts[id].clusterid].count<=max_out)return ;
    if((id-buffers_segment[pts[id].clusterid])<max_out)return;
    double min_distance = DOUBLE_MAX;
    int min_center_index = -1;
    for (size_t j = 0; j < center_num; j++) {
      if(buffers[j].count>=max_out)continue;
      // printf("setAssignmentsMax:%d,%d\n",id,buffers[j].count);
      double distance = calcDist(pts[id], buffers[j]);

      if (distance < min_distance) {
        min_distance = distance;
        min_center_index = j;
      }
    }
    pts[id].clusterid = min_center_index;
    // printf("assignments %d,%d\n",id,min_center_index);
}
__global__ void setAssignments(Pointc * pts,Pointc * centers, int center_num,int size, double DOUBLE_MAX) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >=size) return;
    double min_distance = DOUBLE_MAX;
    int min_center_index = -1;
    for (size_t j = 0; j < center_num; j++) {
      double distance = calcDist(pts[id], centers[j]);
      if (distance < min_distance) {
        min_distance = distance;
        min_center_index = j;
      }
    }
    pts[id].clusterid = min_center_index;
    // printf("assignments %d,%d\n",id,min_center_index);
}
__global__ void removeEmptyStep(Pointc *clusters,Pointc *clusters_empty,int *empty,int size,bool flag){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >=size) return;
  if(!flag)empty[id]=clusters[id].count>0?1:0;
  else {
    if(clusters[id].count>0)clusters_empty[empty[id]]=clusters[id];
  }
}
__global__ void setnewCenter(Pointc * centers,Pointc * new_centers,Pointc* pts,int center_num,int num_instances){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >=center_num) return;
    // printf("setnewCenter %d,%d\n",id,new_centers[id].count);
    if (new_centers[id].count > 0) {
        new_centers[id].x=new_centers[id].x/new_centers[id].count;
        new_centers[id].y=new_centers[id].y/new_centers[id].count;
        centers[id].x=new_centers[id].x;
        centers[id].y=new_centers[id].y;
        // centers[id].cap=new_centers[id].cap/new_centers[id].count;
      } else {
        new_centers[id].cap=0;
        curandState state;
        curand_init(0, id, 0, &state);  // 用不同的序列号初始化
        float rand_val = curand_uniform(&state);
        int index = static_cast<int>(rand_val * num_instances);
        centers[id] = pts[index];
        // new_centers_cap[id]=pts[index].cap;
      }
}
__global__ void CheckNetLengthStep(Pointc* pts,Pointc* clusters,int *clusters_segment,double *net_len,int *clusrerid,int *flag,double getMinLength,double max_net_length,int k){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >=k) return;
  if(flag[id]==1)return;
  double min_x = pts[clusters_segment[id]].x;  
  double min_y = pts[clusters_segment[id]].y;
  double max_x = pts[clusters_segment[id]].x;
  double max_y = pts[clusters_segment[id]].y;
  for(int i=clusters_segment[id];i<clusters_segment[id]+clusters[id].count;i++){
    min_x = fmin(min_x, pts[i].x);
    min_y = fmin(min_y, pts[i].y);
    max_x = fmax(max_x, pts[i].x);
    max_y = fmax(max_y, pts[i].y);
  }
  auto hpmd = (max_x - min_x + max_y - min_y);
  auto hpwl = hpmd / getDbUnit;
  // printf("CheckNetLengthStep:%f\n",net_len[id]);
  if(hpwl < getMinLength || net_len[id] < max_net_length)flag[id]=1;
  else flag[id]=::min((int)std::ceil(net_len[clusrerid[id]] / max_net_length),clusters[id].count);
}
__global__ void variancestep(double *data,double *mean,int num){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >=num) return;
  // printf("mean:%f\n",*mean);
  data[id]=(data[id]-*mean/num)*(data[id]-*mean/num);
}
__global__ void updateMcfvar(Pointc *best_clusters,Pointc *clusters,Pointc* best_pts,Pointc* pts,double *V,int cluster_num,int num_instances){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=num_instances)return;
  best_pts[id]=pts[id];
  if(id>=cluster_num)return;
  best_clusters[id]=clusters[id];
}
__global__ void initAuctionToCluster(Pointc *pts,int*assignments,int max_fanout,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=size)return ;
  pts[id].clusterid=assignments[id]/max_fanout;
}
__global__ void updateKmeansvar(Pointc *buffers,Pointc *clusters,double *V,int cluster_num){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=cluster_num)return;
  buffers[id]=clusters[id];
}
__global__ void getCentroidBuffersStep(segment *seg,Pointc *pts,Point* buffers,int num){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >=num) return;
  double x=0,y=0;
  // x = std::accumulate(cluster.begin(), cluster.end(), x, [](int64_t total, const Inst* inst) { return total + inst->get_location().x(); });
  // y = std::accumulate(cluster.begin(), cluster.end(), y, [](int64_t total, const Inst* inst) { return total + inst->get_location().y(); });
  // return Point(x / cluster.size(), y / cluster.size());
  for(int i=seg[id].l;i<=seg[id].r;i++){x+=pts[i].x,y+=pts[i].y;}
  buffers[id].__x=x/(seg[id].r-seg[id].l+1);
  buffers[id].__y=y/(seg[id].r-seg[id].l+1);
}
__global__ void updateCentersKernel(Pointc* pts, Pointc* new_centers,double* cluster_cap,int num_instances ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_instances) return;
    int center_index = pts[i].clusterid;
    double x = pts[i].x;
    double y = pts[i].y;
    double cap = pts[i].cap;

    // 使用原子操作避免竞争条件
    atomicAdd(&new_centers[center_index].x, x);
    atomicAdd(&new_centers[center_index].y, y);
    atomicAdd(&new_centers[center_index].count, 1);
    atomicAdd(&cluster_cap[center_index], cap);
}

__global__ void updateCluster(double* cap_variance,Pointc* best_clusters,Pointc *clusters,Pointc* pts,Pointc* best_pts,int cluster_num,int num_instances){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id==0){
    // printf("cap_variance:%f,%f\n",cap_variance[0],cap_variance[1]);
    if(cap_variance[0]<cap_variance[1])cap_variance[1]=cap_variance[0];
  }
  __syncthreads(); 
  if(cap_variance[0]>cap_variance[1])return;
  // if(id>=num_instances)return;
  // best_pts[id]=pts[id];
  // if(id>=cluster_num)return;
  for(int i=id;i<cluster_num;i+=(blockDim.x * gridDim.x))best_clusters[i]=clusters[i];
}
__global__ void initNewcenter(Pointc *new_centers,int num){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=num)return;
  new_centers[id].count=0;
}
__global__ void setCenter(Pointc *new_centers,int num){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=num)return;
  new_centers[id].x/=new_centers[id].count;
  new_centers[id].y/=new_centers[id].count;
}
__global__ void initselectNet(Pointc *clusters,int*ptsid,int*clusrerid,int*flag,int net_num){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=net_num)return;
  if(flag[id]==1)ptsid[id]=0,clusrerid[id]=0;
  else ptsid[id]=clusters[id].count,clusrerid[id]=1;
}
__global__ void setselectNet(Pointc *pts,Pointc *select_pts,Pointc* clusters,Pointc* select_clusters,int *clusters_segment,int*ptsid,int*clusrerid,int*flag,int net_num){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=net_num)return;
  // printf("setselectNet:%d %d\n",flag[id],clusters[id].count);
  if(flag[id]==1)return;
  select_clusters[clusrerid[id]]=clusters[id];
  for(int i=0;i<clusters[id].count;i++)select_pts[ptsid[id]+i]=pts[clusters_segment[id]+i];
}
void GPUClustering::calculatesum(Pointc* pts, Pointc* new_pts,int*assignments,int k,int num_instances) {
    // int *d_keys_in;         // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
    // Pointc  *d_values_in;       // e.g., [0, 7, 1, 6, 2, 5, 3, 4]
    int     *d_unique_out;      // e.g., [-, -, -, -, -, -, -, -]
    // Pointc  *d_aggregates_out;  // e.g., [-, -, -, -, -, -, -, -]
    int     *d_num_runs_out;    // e.g., [-]
    sumAll   sum;
    cudaMalloc(&d_unique_out, k*sizeof(int));
    cudaMalloc(&d_num_runs_out, sizeof(int));
    // coutInt(assignments,num_instances);
    // Determine temporary device storage requirements
    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(
      d_temp_storage, temp_storage_bytes,
      assignments, d_unique_out, pts,
      new_pts, d_num_runs_out, sum, num_instances);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run reduce-by-key
    cub::DeviceReduce::ReduceByKey(
      d_temp_storage, temp_storage_bytes,
      assignments, d_unique_out, pts,
      new_pts, d_num_runs_out, sum, num_instances);
    // coutInt(d_unique_out,k);
    cudaFree(d_unique_out);
    cudaFree(d_num_runs_out);
    // coutPointc(new_pts,k);
}
void find_argmax_cub(double* d_array, cub::KeyValuePair<int, double> *d_argmax, int size) {
    // 声明临时存储
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // 获取临时存储需求
    cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_array, d_argmax, size);
    
    // 分配临时存储
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // 执行归约
    cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_array, d_argmax, size);
    
    // 清理
    cudaFree(d_temp_storage);
}
// void GPUClustering::getCentroidBuffers(Pointc* clusters,Pointc*pts,Pointc* buffers,int num_instances){
//   // std::ranges::for_each(clusters, [&buffers](const std::vector<Inst*>& cluster) {
//   //   auto centroid = calcCentroid(cluster);
//   //   buffers.push_back(centroid);
//   // });
//   // return buffers;
//   Pointc* pts;
//   Pointc* h_pts=new Pointc[num_instances];
//   segment* seg;
//   segment* h_seg=new segment[clusters.size()];
//   cudaMalloc(&pts,num_instances*sizeof(Pointc));
//   cudaMalloc(&seg,clusters.size()*sizeof(segment));
//   int c=0;
//   for(int i=0;i<clusters.size();i++){
//     h_seg[i].l=c;
//       for(int j=0;j<clusters[i].size();j++){
//       h_pts[c++]=Pointc(clusters[i][j]->_location.__x,clusters[i][j]->_location.__y,1,clusters[i][j]->cap);
//     }
//     h_seg[i].r=c-1;
//   }
//   cudaMemcpy(pts,h_pts,num_instances*sizeof(Pointc),cudaMemcpyHostToDevice);
//   cudaMemcpy(seg,h_seg,clusters.size()*sizeof(segment),cudaMemcpyHostToDevice);
//   int blockSize=256;
//   int gridSize = (clusters.size() + blockSize - 1) / blockSize;
//   getCentroidBuffersStep<<<gridSize, blockSize>>>(seg,pts,buffers,clusters.size());
//   std::vector<Point> h_buffers;
//   cudaMemcpy(h_buffers.data(),buffers,clusters.size() * sizeof(Point), cudaMemcpyDeviceToHost);
//   return h_buffers;
// }
void GPUClustering::calcVariance(double* d_data,double* variance, int n){
  // double sum = std::accumulate(std::begin(values), std::end(values), 0.0);
  // double mean = sum / values.size();
  // double variance = std::accumulate(std::begin(values), std::end(values), 0.0,
  //                                   [&mean](double total, const double& val) { return total + std::pow(val - mean, 2); });
  // variance /= values.size();
    double *d_temp = nullptr;
    size_t temp_bytes = 0;
    double *mean;
    cudaMalloc(&mean, sizeof(double));
    // 1. Compute mean using CUB's reduce
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_data, mean, n);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_data, mean, n);
    // 2. Compute (x - mean)^2
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    variancestep<<<gridSize, blockSize>>>(d_data,mean, n);
    // 3. Compute variance using CUB's reduce
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_data, variance, n);
    cudaFree(d_temp);
}

void GPUClustering::calcCapVariance(Pointc* clusters,int* clusters_segment,int cluster_num,double* net_len,double* cluster_cap,double *V){//暂时设定
  // cap variance
  // std::vector<double> cluster_cap(clusters.size(), 0);
  thrust::transform(thrust::device,thrust::make_counting_iterator(0),thrust::make_counting_iterator(cluster_num), cluster_cap, 
    [clusters, net_len] __device__ (int i) {return clusters[i].cap + getUnitCap*net_len[i]; });
  // for (size_t i = 0; i < clusters.size(); ++i) {
  //   auto cluster = clusters[i];
  //   // cluster_cap[i] = estimateNetCap(cluster);//暂时不写
  //   cluster_cap[i]=clusters.size()*0.03;
  // }
  // unify cluster cap
  // auto max_val = std::max_element(std::begin(cluster_cap), std::end(cluster_cap));
  // std::ranges::for_each(cluster_cap, [&max_val](double& val) { val /= *max_val; });
  // std::cout<<"calcCapVariance\n";
  // coutDouble(cluster_cap,cluster_num);
  calcVariance(cluster_cap,V,cluster_num);
}
void Maxcub(double* data,double  *d_max,int num){
  void     *d_temp_storage = nullptr;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, data, d_max, num);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, data, d_max, num);
}
void Normalization(double* data,int num){
  double  *d_max; 
  Maxcub(data,d_max,num);
  thrust::transform(thrust::device,data,data+num, data, [d_max] __device__ (const double& p) { return p/(*d_max); });
}
void GPUClustering::calcDelayVariance(Pointc* clusters,int* clusters_segment,int cluster_num,double* net_len,double* net_delay,double* cluster_cap,double *V){//暂时设定
  // delay variance
  double* cluster_delay;
  cudaMalloc(&cluster_delay,cluster_num*sizeof(double));
  thrust::transform(thrust::device,thrust::make_counting_iterator(0),thrust::make_counting_iterator(cluster_num), cluster_delay, 
    // [cluster_cap, net_len,net_delay] __device__ (int i) {return cluster_cap[i] * getUnitRes*net_len[i]/2+net_delay[i]; });
    [cluster_cap, net_len,net_delay] __device__ (int i) {return cluster_cap[i] * getUnitRes*net_len[i]/2; });
  // coutDouble(cluster_delay,cluster_num);
  // Normalization(cluster_delay,cluster_num);
  // coutDouble(cluster_delay,cluster_num);
  calcVariance(cluster_delay,V,cluster_num);
}
void GPUClustering::calcBalanceVariance(Pointc* pts,Pointc* clusters,int* clusters_segment,int cluster_num,std::vector<Inst*>& insts,double* V,
                                              const double& cap_coef, const double& delay_coef){//inst
  // coutInt(clusters_segment,cluster_num);
  // coutPointc(pts,insts.size());
  auto tmr = TreeMaker(pts,clusters,clusters_segment,insts,insts.size(),cluster_num);
  double* net_len;
  cudaMalloc(&net_len,cluster_num*sizeof(double));
  cudaMemcpy(net_len,tmr.net_len,cluster_num*sizeof(double),cudaMemcpyDeviceToDevice);
  // coutDouble(tmr.net_len,cluster_num); 
  double* cluster_cap;
  cudaMalloc(&cluster_cap,cluster_num*sizeof(double));   
  // std::cout<<"TreeMaker\n";                                       
  calcCapVariance(clusters,clusters_segment,cluster_num,tmr.net_len,cluster_cap,&V[1]);
  // std::cout<<"calcCapVariance\n";
  calcDelayVariance(clusters, clusters_segment,cluster_num,tmr.net_len,tmr.net_delay,cluster_cap,&V[2]);
  cudaDeviceSynchronize();
  // std::cout<<"calcBalanceVariance\n";
}
int GPUClustering::selectRandom(double *distances,thrust::default_random_engine &gen,int num_instances,int seed){
  // coutDouble(distances,num_instances);
  thrust::device_vector<double> cdf(num_instances);
  // 1. 计算累积分布函数(CDF)
  thrust::transform_inclusive_scan(
      distances, distances+num_instances,
      cdf.begin(),
      thrust::identity<double>(),
      thrust::plus<double>()
  );
  // 2. 生成随机数并选择
  // thrust::default_random_engine gen(seed);
  thrust::uniform_real_distribution<double> uniform(0.0f, cdf.back());
  double rand_val = uniform(gen);

  // 3. 使用二分查找选择索引
  auto iter = thrust::upper_bound(cdf.begin(), cdf.end(), rand_val);
  int selected_index = iter - cdf.begin();
  if(selected_index>=num_instances)selected_index=num_instances-1;
  return selected_index;
}
void GPUClustering::AuctionToCluster(Pointc* pts,Pointc* clusters,Pointc* buffers,int *clusters_segment,int *assignments,int max_fanout,int num_instances){
  int blockSize = 256;
  int gridSize = (num_instances + blockSize - 1) / blockSize;
  initAuctionToCluster<<<gridSize, blockSize>>>(pts,assignments,max_fanout,num_instances);
  thrust::sort(thrust::device,pts, pts+num_instances);
  updateAssignments<<<gridSize, blockSize>>>(pts,assignments,num_instances);
  int k=std::ceil(1.0 * num_instances / max_fanout);
  gridSize = (k + blockSize - 1) / blockSize;
  initNewcenter<<<gridSize, blockSize>>>(clusters,k);
  calculatesum(pts,clusters,assignments,k,num_instances);
  setCenter<<<gridSize, blockSize>>>(clusters,k);
  initClusterSegment(clusters,clusters_segment,k);
  // std::cout<<"AuctionToCluster:\n";
  cudaMemcpy(buffers,clusters,k*sizeof(Pointc),cudaMemcpyDeviceToDevice);
}
void GPUClustering::kMeansPlusSmall(Pointc* pts,Pointc* clusters,Pointc* best_clusters,int* clusters_segment,int* best_clusters_segment,int *k_segment,int *k,int size,int size1,int num_instances){ //size表示组数
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  // std::cout<<"size:"<<size<<'\n';
  // coutPointc(clusters,size);
  // coutInt(clusters_segment,size);
  Pointc *new_centers,*new_pts;
  double *dis;
  int *assignments,*cluster_k_segment,*cnt1;
  cudaMalloc(&dis,num_instances*sizeof(double));
  cudaMalloc(&new_centers,size1*sizeof(Pointc));
  cudaMalloc(&new_pts,num_instances*sizeof(Pointc));
  cudaMalloc(&assignments,num_instances*sizeof(int));
  cudaMalloc(&cluster_k_segment,size1*sizeof(int));
  cudaMalloc(&cnt1,size1*sizeof(int));
  kMeansPlusSmallStep<<<gridSize, blockSize>>>(pts,clusters,best_clusters,clusters_segment,best_clusters_segment,k_segment,k,size,new_centers,new_pts,dis,assignments,cluster_k_segment,cnt1);
  cudaDeviceSynchronize();

}
void GPUClustering::kMeansPlusBig(Pointc* clusters,Pointc* pts,const size_t& k, const int& seed,const size_t& max_iter, const size_t& no_change_stop,int num_instances){ //inst用的是原先接口
  Pointc *centers;
  cudaMalloc(&centers,k*sizeof(Pointc));

  int cnt=0;
  double* distances;
  cudaMalloc(&distances,num_instances*sizeof(double));
  int blockSize = 256;
  int gridSize = (num_instances + blockSize - 1) / blockSize;
  // cub::KeyValuePair<int, double>  *d_argmax;
  // cudaMalloc(&d_argmax, sizeof(cub::KeyValuePair<int, double>));
  
  thrust::default_random_engine gen(seed);
  while (cnt < k) {
    calculateDistances<<<gridSize, blockSize>>>(pts,centers,distances,cnt,num_instances,std::numeric_limits<double>::max(),1,0);
    cudaDeviceSynchronize();
    // coutDouble(distances,num_instances);
    double d_max=thrust::reduce(thrust::device,distances, distances+num_instances,-std::numeric_limits<double>::max(),thrust::maximum<double>());
    calculateDistances<<<gridSize, blockSize>>>(pts,centers,distances,cnt,num_instances,std::numeric_limits<double>::max(),max(1.,d_max),1);
    // coutDouble(distances,num_instances);
    cudaDeviceSynchronize();
    int selected_index=selectRandom(distances,gen,num_instances,cnt);
    cudaMemcpy(&centers[cnt], &pts[selected_index], sizeof(Pointc), cudaMemcpyDeviceToDevice);
    cnt++;
  }
  /*
  std::cout<<"initBegincenter\n";
  initBegincenter<<<gridSize, blockSize>>>(pts,centers,distances,k,num_instances,std::numeric_limits<double>::max());
  */
  // coutPointc(centers,cnt);
  size_t num_iterations = 0;
  // double prev_cap_variance = std::numeric_limits<double>::max();
  size_t no_change = 0;
  Pointc* new_centers,*best_pts;
  double* cluster_cap,* cap_variance;
  cudaMalloc(&cap_variance,2*sizeof(double)); //cur-0,pre-1
  cudaMalloc(&new_centers,k*sizeof(Pointc));
  cudaMalloc(&best_pts,num_instances*sizeof(Pointc));
  cudaMalloc(&cluster_cap,k*sizeof(double));
  thrust::fill(thrust::device,cap_variance, cap_variance + 2, std::numeric_limits<double>::max()); 
  int* assignments;
  cudaMalloc(&assignments,num_instances*sizeof(int));
  while (num_iterations++ < max_iter && no_change++ < no_change_stop) {
    // Assignment step
    gridSize = (num_instances + blockSize - 1) / blockSize;
    setAssignments<<<gridSize, blockSize>>>(pts,centers,k,num_instances,std::numeric_limits<double>::max());
    thrust::sort(thrust::device,pts, pts+num_instances);
    thrust::transform(thrust::device,pts, pts + num_instances, assignments, [] __device__ (const Pointc& p) { return p.clusterid; });
    gridSize = (k + blockSize - 1) / blockSize;
    initNewcenter<<<gridSize, blockSize>>>(new_centers,k);
    calculatesum(pts,new_centers,assignments,k,num_instances);
    // coutPointc(pts,10);
    gridSize = (k + blockSize - 1) / blockSize;
    setnewCenter<<<gridSize, blockSize>>>(centers,new_centers,pts,k,num_instances); //new_center_cap处理
    thrust::transform(thrust::device,new_centers, new_centers+k, cluster_cap, [] __device__ (const Pointc& p) { return p.cap; });
    calcVariance(cluster_cap,cap_variance,k);
    // coutPointc(new_centers,k);
    // updateCluster<<<gridSize, blockSize>>>(cap_variance,clusters,new_centers,pts,best_pts,k,num_instances);
    updateCluster<<<1, blockSize>>>(cap_variance,clusters,new_centers,pts,best_pts,k,num_instances);
    cudaDeviceSynchronize();
  }
  // cudaMemcpy(pts,best_pts,num_instances*sizeof(Pointc),cudaMemcpyDeviceToDevice);
  cudaFree(best_pts);
  cudaFree(cluster_cap);
  cudaFree(centers);
}
void GPUClustering::initClusterSegment(Pointc* clusters,int *clusters_segment,int num){
  // thrust::transform(thrust::device,clusters, clusters + num, clusters_segment,[] __device__ (const Pointc& p) { return p.count; });
  int blockSize = 256;
  int gridSize = (num + blockSize - 1) / blockSize;
  setClusternum<<<gridSize, blockSize>>>(clusters,clusters_segment,num);
  thrust::exclusive_scan(thrust::device,clusters_segment, clusters_segment + num, clusters_segment);
}
void GPUClustering::INIT(int num){
  cudaMemcpyToSymbol(getUnitCap, &h_getUnitCap, sizeof(double));
  cudaMemcpyToSymbol(getUnitRes, &h_getUnitRes, sizeof(double));
  cudaMemcpyToSymbol(getDbUnit, &h_getDbUnit, sizeof(double));
  instsize=num;
}
void GPUClustering::SimpleMCF(Pointc* pts,Pointc* clusters,Pointc* buffers,int *buffers_segment,int max_fanout,int size){
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  int k=std::ceil(1.0 * size / max_fanout);
  int* assignments;
  cudaMalloc(&assignments,size*sizeof(int));
  setAssignments<<<gridSize, blockSize>>>(pts,buffers,k,size,std::numeric_limits<double>::max());
  thrust::sort(thrust::device,pts, pts+size);
  thrust::transform(thrust::device,pts, pts + size, assignments, [] __device__ (const Pointc& p) { return p.clusterid; });
  gridSize = (k + blockSize - 1) / blockSize;
  initNewcenter<<<gridSize, blockSize>>>(buffers,k);
  calculatesum(pts,buffers,assignments,k,size);
  setCenter<<<gridSize, blockSize>>>(buffers,k); 
  initClusterSegment(buffers,buffers_segment,k);
  for(int i=0;i<0;i++){
    setAssignmentsMax<<<gridSize, blockSize>>>(pts,buffers,buffers_segment,max_fanout,k,size,std::numeric_limits<double>::max());
    thrust::sort(thrust::device,pts, pts+size);
    thrust::transform(thrust::device,pts, pts + size, assignments, [] __device__ (const Pointc& p) { return p.clusterid; });
    gridSize = (k + blockSize - 1) / blockSize;
    initNewcenter<<<gridSize, blockSize>>>(buffers,k);
    calculatesum(pts,buffers,assignments,k,size);
    setCenter<<<gridSize, blockSize>>>(buffers,k); 
    initClusterSegment(buffers,buffers_segment,k);
  }
  cudaMemcpy(clusters,buffers,k*sizeof(Pointc),cudaMemcpyDeviceToDevice);
  // std::cout<<"SimpleMCF\n";
  cudaFree(assignments);
}
void GPUClustering::Auction(const std::vector<Inst*>& insts,Pointc* pts,Pointc* clusters,Pointc*buffers,int *clusters_segment,int max_fanout,int cluster_num){
    // std::ofstream outfile("/root/autodl-tmp/keans/data/aution3.txt");
    // Pointc h_buffers[cluster_num];
    // std::cout<<cluster_num<<'\n';
    // cudaMemcpy(h_buffers,buffers,sizeof(Pointc)*cluster_num,cudaMemcpyDeviceToHost);
    // for(int i=0;i<cluster_num;i++)std::cout<<h_buffers[i].x<<' '<<h_buffers[i].y<<'\n';
    // Pointc h_pts[insts.size()];
    // cudaMemcpy(h_pts,pts,sizeof(Pointc)*insts.size(),cudaMemcpyDeviceToHost);
    // for(int i=0;i<insts.size();i++)outfile<<h_pts[i].x<<' '<<h_pts[i].y<<'\n';
    // outfile.close();  
    int blockSize = 256;
    int gridSize = (cluster_num*cluster_num*max_fanout*max_fanout + blockSize - 1) / blockSize;
    double * cost_matrices,*d_max;
    // double *h_max=new double[1];
    int *fdom;
    cudaMalloc(&fdom,1000*sizeof(int));
    auto start = std::chrono::high_resolution_clock::now();
    initfdom<<<1,1>>>(fdom,1000);
    cudaDeviceSynchronize();
    cudaMalloc(&cost_matrices,cluster_num*cluster_num*max_fanout*max_fanout*sizeof(double));
    initCostMat<<<gridSize, blockSize>>>(pts,buffers,cost_matrices,insts.size(),cluster_num*max_fanout,max_fanout,
    cluster_num*cluster_num*max_fanout*max_fanout,0,fdom);
    cudaDeviceSynchronize();
    // Maxcub(cost_matrices,d_max,cluster_num*max_fanout*cluster_num*max_fanout);
    // cudaMemcpy(h_max,d_max,sizeof(double),cudaMemcpyDeviceToHost);
    // initCostMat<<<gridSize, blockSize>>>(pts,buffers,cost_matrices,insts.size(),cluster_num*max_fanout,max_fanout,
    // cluster_num*cluster_num*max_fanout*max_fanout,1,fdom,d_max);
    // cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Auction prepare time: " << elapsed.count() << " seconds" << std::endl;
    AuctionRun(pts,clusters,buffers,clusters_segment,cost_matrices,max_fanout,insts.size());
    cudaFree(cost_matrices);
}
void GPUClustering::selectNet(Pointc *pts,Pointc *select_pts,Pointc* clusters,Pointc* select_clusters,int *clusters_segment,
  int *select_clusters_segment,int *clusterid,int *flag,int num_instances,int &select_num_instances,int cluster_num, int &select_cluster_num){
  int *ptsid;
  cudaMalloc(&ptsid,cluster_num*sizeof(int));
  int blockSize = 256;
  int gridSize = (cluster_num + blockSize - 1) / blockSize;
  initselectNet<<<gridSize, blockSize>>>(clusters,ptsid,clusterid,flag,cluster_num);
  select_num_instances = thrust::reduce(thrust::device,ptsid,ptsid+cluster_num);
  select_cluster_num = thrust::reduce(thrust::device,clusterid,clusterid+cluster_num);
  thrust::exclusive_scan(thrust::device,ptsid, ptsid + cluster_num, ptsid);
  thrust::exclusive_scan(thrust::device,clusterid, clusterid + cluster_num, clusterid);
  // std::cout<<cluster_num<<' '<<select_num_instances<<' '<<select_cluster_num<<'\n';
  setselectNet<<<gridSize, blockSize>>>(pts,select_pts,clusters,select_clusters,clusters_segment,ptsid,clusterid,flag,cluster_num);
  initClusterSegment(select_clusters,select_clusters_segment,select_cluster_num);
}
void GPUClustering::CheckNetLength(Pointc *pts,Pointc* clusters,int *clusters_segment,int *flag,std::vector<Inst*>& insts,double max_len,int cluster_num){
  int blockSize = 256;
  int gridSize = (cluster_num + blockSize - 1) / blockSize;
  // auto bst = BoundSkewTree(pts,clusters,clusters_segment,num_instances,cluster_num);
  // bst.run();
  // bst.recursiveBottomUp();
  // bst.embedding();
  // bst.convert();
  // bst.calcNetLen();  
  Pointc* select_pts,*select_clusters;
  int *select_clusters_segment,*clusterid,select_num_instances,select_cluster_num;
  cudaMalloc(&select_pts,insts.size()*sizeof(Pointc));
  cudaMalloc(&select_clusters,cluster_num*sizeof(Pointc));
  cudaMalloc(&select_clusters_segment,cluster_num*sizeof(int));
  cudaMalloc(&clusterid,cluster_num*sizeof(int));
  selectNet(pts,select_pts,clusters,select_clusters,clusters_segment,select_clusters_segment,clusterid,flag,insts.size(),select_num_instances,cluster_num,select_cluster_num);
  // std::cout<<"selecNet\n"<<select_cluster_num<<'\n';
  auto tmr = TreeMaker(select_pts,select_clusters,select_clusters_segment,insts,select_num_instances,select_cluster_num); //select_insts
  // std::cout<<"CheckNetLength\n";
  CheckNetLengthStep<<<gridSize, blockSize>>>(pts,clusters,clusters_segment,tmr.net_len,clusterid,flag,50.,max_len,cluster_num);
  cudaDeviceSynchronize();
}
int GPUClustering::removeEmpty(Pointc*clusters,int size){
  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;
  int *empty;
  Pointc *clusters_empty;
  cudaMalloc(&empty,size*sizeof(int));
  cudaMalloc(&clusters_empty,size*sizeof(Pointc));
  removeEmptyStep<<<gridSize, blockSize>>>(clusters,clusters_empty,empty,size,0);
  int cluster_num = thrust::reduce(thrust::device,empty, empty + size);
  thrust::exclusive_scan(thrust::device,empty, empty + size, empty);
  removeEmptyStep<<<gridSize, blockSize>>>(clusters,clusters_empty,empty,size,1);
  cudaMemcpy(clusters,clusters_empty,cluster_num*sizeof(Pointc),cudaMemcpyDeviceToDevice);
  return cluster_num;
}
int GPUClustering::slackClustering(Pointc* pts,Pointc*clusters,int *clusters_segment,std::vector<Inst*>& insts,double max_len,int k){
  initClusterSegment(clusters,clusters_segment,k);
  int *flag;
  cudaMalloc(&flag,insts.size()*sizeof(int));
  cudaMemset(flag, 0, insts.size() * sizeof(int));
  int *k_segment,*best_clusters_segment;
  cudaMalloc(&k_segment,insts.size()*sizeof(int));
  cudaMalloc(&best_clusters_segment,insts.size()*sizeof(int));
  Pointc *best_clusters;
  cudaMalloc(&best_clusters,insts.size()*sizeof(Pointc));
  int cluster_num=k;
  for(int iter=0;iter<1;iter++){
    // coutPointc(clusters,cluster_num);
    cluster_num=removeEmpty(clusters,cluster_num);
    // std::cout<<"removeEmpty\n";
    initClusterSegment(clusters,clusters_segment,cluster_num);
    CheckNetLength(pts,clusters,clusters_segment,flag,insts,max_len,cluster_num);
    int cluster_num1 = thrust::reduce(thrust::device,flag, flag + cluster_num);
    thrust::exclusive_scan(thrust::device,flag, flag + cluster_num, k_segment);
    // coutInt(k_segment,cluster_num);
    // coutPointc(clusters,cluster_num);
    kMeansPlusSmall(pts,clusters,best_clusters,clusters_segment,best_clusters_segment,k_segment,flag,cluster_num,cluster_num1,insts.size());
    // std::cout<<cluster_num1<<'\n';
    // coutPointc(best_clusters,cluster_num1);
    cudaMemcpy(clusters,best_clusters,cluster_num1*sizeof(Pointc),cudaMemcpyDeviceToDevice);
    cluster_num=cluster_num1;
    initClusterSegment(clusters,clusters_segment,cluster_num);
    // coutInt(clusters_segment,cluster_num);
  }
  cluster_num=removeEmpty(clusters,cluster_num);
  initClusterSegment(clusters,clusters_segment,cluster_num);
  return cluster_num;
}
int GPUClustering::iterClustering(std::vector<Inst*>& insts,Pointc* pts,Pointc* clusters, const size_t& max_fanout,
                           const size_t& iters, const size_t& no_change_stop,const double& limit_ratio){
  
  // LOG_FATAL_IF(max_fanout < 2) << "max_fanout should be greater than 1";
  if (insts.size() == 2) {
    // return {insts};
    return 1;
  }
  if(insts.size()>20000){
    auto divide_clusters =kMeans(insts,4, 0, 100);
    // for(int i=0;i<divide_clusters.size();i++){
    //   std::cout<<divide_clusters[i].size()<<'\n';
    // }
    // std::cout<<"kmeans\n";
    int sum=0,cluster_sum=0;
    for(int i=0;i<divide_clusters.size();i++){
      Pointc* ptsi=pts+sum,* clustersi=clusters+cluster_sum;
      int cluster_num=iterClustering(divide_clusters[i],ptsi,clustersi,32,7,5,1);
      sum+=divide_clusters[i].size();
      cluster_sum+=cluster_num;
    }
    return cluster_sum;
  }
  // LOG_INFO_IF(log) << "iterative clustering";
  // initialize clusters
  size_t cluster_num = std::ceil(1.0 * insts.size() / (limit_ratio * max_fanout));
  if (cluster_num == insts.size()) {
    cluster_num = insts.size() - 1;
  }

  int *clusters_segment;
  cudaMalloc(&clusters_segment,cluster_num*sizeof(int));
  Pointc* best_clusters,*best_pts;
  cudaMalloc(&best_clusters,cluster_num*sizeof(Pointc));
  cudaMalloc(&best_pts,insts.size()*sizeof(Pointc));
  auto startk = std::chrono::high_resolution_clock::now();
  kMeansPlusBig(clusters,pts,cluster_num,0,100,5, insts.size());
  initClusterSegment(clusters,clusters_segment,cluster_num);
  // coutInt(clusters_segment,cluster_num);
  auto endk = std::chrono::high_resolution_clock::now();//auction
    std::chrono::duration<double> elapsedk = endk - startk;
    // std::cout << "kMeansPlusBig execution time: " << elapsedk.count() << " seconds" << std::endl;
  // coutInt(clusters_segment,cluster_num);
  // if (clusters.size() == 1) {
  //   return clusters;
  // }
  // make temp centroid buffer
  Pointc* buffers;
  cudaMalloc(&buffers,cluster_num*sizeof(Pointc));
  cudaMemcpy(buffers,clusters,cluster_num*sizeof(Pointc),cudaMemcpyDeviceToDevice);
  // // cudaMalloc(d_clusters,insts.size()*sizeof(Point));
  // // getCentroidBuffers(clusters,buffers,num_instances);
  // cudaMemcpy(buffers,clusters,k*sizeof(Pointc),cudaMemcpyDeviceToDevice);
  // auto best_clusters = clusters;
  // auto kmeans_var = calcBalanceVariance(clusters, buffers,cluster_num);
  double *mcf_var;
  cudaMalloc(&mcf_var,6*sizeof(double));//0-mcf_var,1-new_mcf_var_cap,2-new_mcf_var_delay
  thrust::fill(thrust::device,mcf_var, mcf_var + 6, std::numeric_limits<double>::max()); 
  // auto mcf_var = std::numeric_limits<double>::max();
  // // LOG_INFO_IF(log) << "initial kmeans var: " << kmeans_var;
  size_t no_change = 0;
  // // iterate
  double h_mcf_var[6];
  h_mcf_var[0]=std::numeric_limits<double>::max();
  h_mcf_var[3]=std::numeric_limits<double>::max();
  for (size_t i = 0; i < iters; ++i) {
    if ((i + 1) % 50 == 0) {
      // LOG_INFO_IF(log) << "iter: " << i + 1;
    }
    // std::cout<<i<<"k\n";
    // coutPointc(pts,10);
    auto start1 = std::chrono::high_resolution_clock::now();
    AllocateRun(insts,pts,clusters,buffers,max_fanout,cluster_num);
    initClusterSegment(clusters,clusters_segment,cluster_num);
    // std::cout<<"initClusterSegment\n";coutInt(clusters_segment,cluster_num);
    // coutCost(pts,clusters_segment,buffers,cluster_num,insts.size());
    cudaMemcpy(buffers,clusters,cluster_num*sizeof(Pointc),cudaMemcpyDeviceToDevice);
    // // SimpleMCF(pts,clusters,buffers,clusters_segment,max_fanout,insts.size());
    int blockSize = 256;
    int gridSize = (cluster_num*cluster_num*max_fanout*max_fanout + blockSize - 1) / blockSize;
    auto start = std::chrono::high_resolution_clock::now();//auction
    std::chrono::duration<double> elapsed1 = start- start1 ; 
    // std::cout << "MinCostFlowRun execution time: " << elapsed1.count() << " seconds" << std::endl;
    // Pointc* b;
    // cudaMalloc(&b,cluster_num*sizeof(Pointc));
    // cudaMemcpy(b,buffers,cluster_num*sizeof(Pointc),cudaMemcpyDeviceToDevice);
    // Auction(insts,pts,clusters,buffers,clusters_segment,max_fanout,cluster_num);
    // coutCost(pts,clusters_segment,b,cluster_num,insts.size());
    // cudaFree(b);
    // coutInt(clusters_segment,cluster_num);
    auto end = std::chrono::high_resolution_clock::now();//auction
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Auction execution time: " << elapsed.count() << " seconds" << std::endl;
    if(i<2)continue;
    calcBalanceVariance(pts,clusters,clusters_segment,cluster_num,insts,mcf_var);
    gridSize = (insts.size() + blockSize - 1) / blockSize;
    cudaMemcpy(h_mcf_var+1,mcf_var+1,2*sizeof(double),cudaMemcpyDeviceToHost);
    bool need_kmeans_update=true;
    if((h_mcf_var[1]+h_mcf_var[2]) < h_mcf_var[0]){
      updateMcfvar<<<gridSize, blockSize>>>(best_clusters,clusters,best_pts,pts,mcf_var,cluster_num,insts.size());
      h_mcf_var[0]=h_mcf_var[1]+h_mcf_var[2];
      need_kmeans_update=false;
    }
    if (need_kmeans_update) {
      // std::cout<<"need_kmeans_update\n";
      kMeansPlusBig(clusters,pts,cluster_num,i,50,5, insts.size());
      // coutPointc(clusters,cluster_num);
      initClusterSegment(clusters,clusters_segment,cluster_num);
    //   coutInt(clusters_segment,cluster_num);
      // calcBalanceVariance(pts,clusters,clusters_segment,cluster_num,insts,&mcf_var[3]);
      // gridSize = (cluster_num + blockSize - 1) / blockSize;
      // cudaMemcpy(h_mcf_var+4,mcf_var+4,2*sizeof(double),cudaMemcpyDeviceToHost);
      // std::cout<<h_mcf_var[3]<<' '<<h_mcf_var[4]<<' '<<h_mcf_var[5]<<'\n';
      // if((h_mcf_var[4]+h_mcf_var[5]) < h_mcf_var[3]){
        cudaMemcpy(buffers,clusters,cluster_num*sizeof(Pointc),cudaMemcpyDeviceToDevice);
    //     h_mcf_var[3]=h_mcf_var[4]+h_mcf_var[5];
    //   }
    }
  }
  cudaMemcpy(clusters,best_clusters,cluster_num*sizeof(Pointc),cudaMemcpyDeviceToDevice);
  cudaMemcpy(pts,best_pts,insts.size()*sizeof(Pointc),cudaMemcpyDeviceToDevice);
  cudaFree(best_clusters);
  cudaFree(best_pts);
  cudaFree(clusters_segment);
  return cluster_num;
}


void GPUClustering::coutCost(Pointc* pts, int* clusters_segment, Pointc* buffers,int cluster_num,int size){
  Pointc h_pts[size],h_buffers[cluster_num];
  // int h_clusters_segment[cluster_num];
  int *h_clusters_segment = new int[cluster_num];
  cudaMemcpy(h_pts, pts, (size) * sizeof(Pointc), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_buffers, buffers, cluster_num * sizeof(Pointc), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_clusters_segment, clusters_segment, cluster_num * sizeof(int), cudaMemcpyDeviceToHost);
  double cost=0;
  for(int i=0;i<cluster_num;i++){
    int j=h_clusters_segment[i];
    while(((i+1)<cluster_num&&j<h_clusters_segment[i+1])||((i+1)==cluster_num&&j<size)){
      cost=cost+std::fabs(h_pts[j].x-h_buffers[i].x)+std::fabs(h_pts[j].y-h_buffers[i].y);
      j++;
    }
  }
  std::cout<<"cost:"<<cost<<'\n';
  delete []h_clusters_segment;
}
void GPUClustering::coutPointc(Pointc* d,int size){
  Pointc *h = new Pointc[size];
  cudaMemcpy(h, d, (size) * sizeof(Pointc), cudaMemcpyDeviceToHost);
  std::cout << "Pointc: ";
  for (int i = 0; i < size ; i++) {
    // if(h[i]>100000)std::quick_exit(EXIT_FAILURE); 
      std::cout << h[i].x << " "<<h[i].y<<' '<<h[i].cap<<' '<<h[i].count<<' '<<h[i].instID<<'\n';
  }
  std::cout << std::endl;
  delete []h;
}
void GPUClustering::coutDouble(double* d,int size){
  double *h = new double[size];
  cudaMemcpy(h, d, (size) * sizeof(double), cudaMemcpyDeviceToHost);
  std::cout << "values: ";
  for (int i = 0; i < size ; i++) {
      std::cout << h[i] << " ";
  }
  std::cout << std::endl;
}
void GPUClustering::coutInt(int* d,int size){
  int *h = new int[size];
  cudaMemcpy(h, d, (size) * sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "values: ";
  for (int i = 0; i < size ; i++) {
      std::cout << h[i] << " ";
  }
  std::cout << std::endl;
}
}