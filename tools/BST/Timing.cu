#include <cub/cub.cuh>
#include "BoundSkewTree.h"
namespace gcts {

__device__ double calcDist(Pt p1,Pt p2){
  return fabs(p1.x - p2.x) + fabs(p1.y - p2.y);
}  
__global__ void calcNetLenStep(Area* areas,double *net_len,int* stack,int pinsize,int netnum){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=netnum)return ;
  // int stack[128];
  // printf("%d\n",id);
  int cnt=0,begin=areas[id+pinsize].l;
  stack[begin+(cnt++)]=id+pinsize;
  double length=0;
  // if(id==0)printf("%d:[%d,%d]\n",id,areas[id+pinsize].l,areas[id+pinsize].r);
  if(areas[id+pinsize].l>=areas[id+pinsize].r){
    net_len[id]=length;
    return ;
  }
  while(cnt>0){
    // printf("calcNetLenStep:%d %d %f %d\n",id,stack[cnt-1],areas[stack[cnt-1]].snake,areas[stack[cnt-1]].p);
    auto cur=areas[stack[begin+(--cnt)]];
    // printf("%d\n",begin+cnt);
    length+=cur.snake;
    if(cur.p!=-1){
      length+=calcDist(cur.location,areas[cur.p].location);
      // printf("calcNetLenStep:%d %f %f %f %f %f\n",id,length,cur.location.x,areas[cur.p].location.x,cur.location.y,areas[cur.p].location.y);
    }
    // printf("calcNetLenStep:%d %d %d\n",cnt,cur.l,cur.r);
    if(cur.l==cur.r)continue;
    // if(cnt>pinsize)printf("!!!");
    if(cur.left!=-1)stack[begin+(cnt++)]=cur.left;
    if(cur.right!=-1)stack[begin+(cnt++)]=cur.right;
  }
  // printf("%d: length:%f\n",id,length);
  net_len[id]=length;
}
__global__ void NetDelayStep(Area* areas,double *net_delay,int pinsize,int netnum){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=netnum)return ;
  net_delay[id]=areas[id+pinsize].location.max;
  // printf("NetDelayStep:%f\n",net_delay[id]);
}
void BoundSkewTree::calcNetLen(){
  int blockSize = 256;
  int gridSize = (net_num + blockSize - 1) / blockSize;
  if(net_len==nullptr)cudaMalloc(&net_len ,net_num* sizeof(double));
  int *stack=nullptr;
  cudaMalloc(&stack,pinsize*sizeof(int));
  cudaError_t err = cudaGetLastError();
  calcNetLenStep<<<gridSize, blockSize>>>(areas,net_len,stack,pinsize,net_num);
  if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
  cudaFree(stack);
}
void BoundSkewTree::estimateNetDelay(){
  int blockSize = 256;
  int gridSize = (net_num + blockSize - 1) / blockSize;
  if(net_delay==nullptr)cudaMalloc(&net_delay,net_num*sizeof(double));
  NetDelayStep<<<gridSize, blockSize>>>(areas,net_delay,pinsize,net_num);
}
}