#include "LevelSolve.h"
#include "Timing.h"
namespace gcts {
__device__ double calcDist(Pointc& p1,Pointc& p2){
  return std::fabs(p1.x - p2.x) + std::fabs(p1.y - p2.y);
}
__global__ void LCAquery(Area* areas,Pointc* pts,Pointc* clusters,int* levelsums,int *ancestors,int *segment,Pointc* guide_locs,int maxlevel,int *key,int size){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id>=size)return;
    int l=segment[id],r=segment[id]+clusters[id].count;
    int minlevel=0,minid;
    // printf("id: %d %d %d\n",id,l,r);
    for(int i=l;i<r;i++){
      // printf("%d:%f %f|\n",pts[i].instID,pts[i].x,pts[i].y);
        // printf("%d:%f %f |  %d %f %f\n",pts[i].instID,pts[i].x,pts[i].y,key[pts[i].instID],areas[key[pts[i].instID]].x,areas[key[pts[i].instID]].y);
        ancestors[i]=areas[key[pts[i].instID]].p;
        if(i==l)minid=ancestors[i];
        else minid=min(ancestors[i],minid);
    }
    // printf("mid %d\n",minid);
    for(int i=maxlevel;i>0;i--){
        if(minid<levelsums[i]&&minid>=levelsums[i-1]){minlevel=i-1;break;}
    }
    for(int i=l;i<r;i++){
        while(true){
            if(areas[ancestors[i]].p<levelsums[minlevel])break;
            ancestors[i]=areas[ancestors[i]].p;
        }
    }
    int ans;
    while(true){
        bool flag=true;  
        ans=ancestors[l];
        for(int i=l+1;i<r;i++)if(ans!=ancestors[i]){flag=false;}
        if(flag)break;
        for(int i=l;i<r;i++)ancestors[i]=areas[ancestors[i]].p;
    }
    guide_locs[id]=Pointc(areas[ans].x,areas[ans].y);
    double x=0,y=0;
    for(int i=l;i<r;i++){x+=pts[i].x,y+=pts[i].y;}
    Pointc P=Pointc(x/clusters[id].count,y/clusters[id].count);
    while(areas[ans].p!=-1&&calcDist(guide_locs[id],P)<0.005){
      ans=areas[ans].p;
      guide_locs[id]=Pointc(areas[ans].x,areas[ans].y);
    }
    // printf("[%d,%d],guide_locs(%d):%f,%f\n",l,r-1,ans,guide_locs[id].x,guide_locs[id].y);
}
__global__ void shiftupdate(Pointc *guide_locs,Pointc *pts, Pointc* clusters,int *clusters_segment,double *net_dist,int max_dist,int level, int cluster_num){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=cluster_num)return;
  // printf("shiftupdate:%d\n",(int)net_dist[id]);
  if ((int)net_dist[id] <= max_dist) {
    auto center = Pointc(clusters[id].x/clusters[id].count,clusters[id].y/clusters[id].count);
    int center_dist = std::ceil(calcDist(center,guide_locs[id]));
    auto ratio = 0.1 + (level - 1) * 0.25;
    ratio = ratio > 1 ? 1 : ratio;
    int allow_center_dist = ratio * center_dist;
    auto shift_dist = min(max_dist - (int)net_dist[id], allow_center_dist);
    guide_locs[id] = center_dist > 0 ? (guide_locs[id] - center) * (1.0 * shift_dist / center_dist) + center : center;
  }
}
__global__ void initkey(Area*areas,int *key,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=size)return;
  // printf("key id:%d %d\n",id,areas[id].ptsid);
  key[areas[id].ptsid]=id;
}
void LevelSolve::guideCenter(Pointc* pts,Pointc* clusters,Pointc* guide_locs,int* clusters_segment,std::vector<Inst*>& insts,int instsize,int cluster_num){
    // std::cout<<"guideCenter\n";

    int num_instances=insts.size();
    Pointc *cpts=new Pointc[num_instances];
    Pointc *tpts;
    int cnt=0;
    for(int i=0;i<num_instances;i++){
      cpts[i]=Pointc(insts[i]->_location.x(),insts[i]->_location.y());
      cpts[i].cap=insts[i]->cap;
      cpts[i].instID=i;
    }
    cudaMalloc(&tpts,num_instances*sizeof(Pointc));
    cudaMemcpy(tpts, cpts, num_instances*sizeof(Pointc), cudaMemcpyHostToDevice);
    auto tmr = TreeMaker(tpts,insts,instsize);
    // std::cout<<"guideCenter end\n";
    int *ancestors;
    cudaMalloc(&ancestors,instsize*sizeof(int));
    cudaDeviceSynchronize();
    LCA(tmr.areas,pts,clusters,tmr.levelsum,clusters_segment,ancestors,tmr.maxlevel,guide_locs,instsize,cluster_num);
    delete []cpts;
    cudaFree(tpts);
}
void LevelSolve::LCA(Area* areas,Pointc* pts,Pointc* clusters,int *levelsum,int *clusters_segment,int *ancestors,int maxlevel,Pointc* guide_locs,int instsize,int cluster_num){
    int blockSize = 256;
    int gridSize = (cluster_num + blockSize - 1) / blockSize;
    int *levelsums;
    cudaMalloc(&levelsums,(maxlevel+1)*sizeof(int));
    cudaMemcpy(levelsums,levelsum,(maxlevel+1)*sizeof(int),cudaMemcpyHostToDevice);
    int *key;
    cudaMalloc(&key,instsize*sizeof(int));
    gridSize = (instsize + blockSize - 1) / blockSize;
    initkey<<<gridSize, blockSize>>>(areas,key,instsize);
    cudaDeviceSynchronize();
    LCAquery<<<gridSize, blockSize>>>(areas,pts,clusters,levelsums,ancestors,clusters_segment,guide_locs,maxlevel,key,cluster_num);
    cudaDeviceSynchronize();
    cudaFree(key);
}
// void LevelSolve::levelProcess(Pointc *guide_locs,Pointc *pts, Pointc* clusters,int *clusters_segment,std::vector<Inst*>& insts,int cluster_num){
//   // int net_dist = BalanceClustering::estimateNetLength(load_pins) * TimingPropagator::getDbUnit();
//   auto tmr = TreeMaker(pts,clusters,clusters_segment,insts,insts.size(),cluster_num);
//   // int blockSize = 256;
//   // int gridSize = (cluster_num + blockSize - 1) / blockSize;
// }
void LevelSolve::CPUtoGPU(std::vector<Inst*>& insts){
  int num_instances=insts.size();
  Pointc *cpts=new Pointc[num_instances];
  int cnt=0;
  for(int i=0;i<num_instances;i++){
    cpts[i]=Pointc(insts[i]->_location.x(),insts[i]->_location.y());
    cpts[i].cap=insts[i]->cap;
    cpts[i].instID=i;
  }
  cudaMalloc(&pts,num_instances*sizeof(Pointc));
  cudaMemcpy(pts, cpts, num_instances*sizeof(Pointc), cudaMemcpyHostToDevice);
  cudaMalloc(&clusters,num_instances*sizeof(Pointc));
  cudaMalloc(&clusters_segment,num_instances*sizeof(int));
}
void LevelSolve::GPUtoCPU(std::vector<Inst*>& insts,Pointc* guide_locs,int cluster_num){
  Pointc h_guide_locs[cluster_num];
  cudaMemcpy(h_guide_locs, guide_locs, cluster_num*sizeof(Pointc), cudaMemcpyDeviceToHost);
  for(int i=0;i<cluster_num;i++){
    Point location(h_guide_locs[i].x, h_guide_locs[i].y);
    // std::cout<<h_guide_locs[i].x<<' '<<h_guide_locs[i].y<<'\n';
    Inst* inst;
    if(Timing::bufferlibs.size()>0)inst = new Inst(Timing::bufferlibs[0], location);
    else inst = new Inst("", location);
    insts.push_back(inst);
  }
}
void check(std::vector<Inst*>& insts){
  std::unordered_map<std::string, int> map;
  for(int i=0;i<insts.size();i++){
    int xx=Timing::_db_unit*insts[i]->get_location().x(),yy=Timing::_db_unit*insts[i]->get_location().y();
    // std::cout<<xx<<' '<<yy<<'\n';
    std::string s=std::to_string(xx)+std::to_string(yy);
    if(map.count(s)==0)map[s]=i;
    else{
      while(map.count(s)!=0){
        xx+=2;
        s=std::to_string(xx)+std::to_string(yy);
      }
    }
    map[s]=i;
    insts[i]->set_location(Point(xx*1./Timing::_db_unit,yy*1./Timing::_db_unit));
    insts[i]->ID=i;
  }
}

std::vector<Inst*> LevelSolve::assignApply(std::vector<Inst*>& insts,int max_fanout,double max_len,int level){
  if(Timing::_db_unit==0)Timing::init();
  CPUtoGPU(insts);
  auto clu=GPUClustering();
  clu.INIT(insts.size());
  auto start = std::chrono::high_resolution_clock::now();
  // clu.coutPointc(pts,insts.size());
  int cluster_num = clu.iterClustering(insts,pts,clusters,max_fanout,5,5,1);
  // std::cout<<"iterClustering:"<<cluster_num<<"\n";
  // clu.coutPointc(pts,insts.size());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  // std::cout << "iterClustering execution time: " << elapsed.count() << " seconds" << std::endl;
  // std::cout<<"iterClustering:"<<cluster_num<<'\n';
  cluster_num = clu.slackClustering(pts,clusters,clusters_segment,insts,max_len,cluster_num);
  // std::cout<<"slackClustering:"<<cluster_num<<'\n';
  auto end1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed1 = end1 - end;
  // std::cout << "slackClustering execution time: " << elapsed1.count() << " seconds" << std::endl;
  Pointc *guide_locs;
  // clu.coutPointc(pts,insts.size());
  cudaMalloc(&guide_locs,cluster_num*sizeof(Pointc));
  // clu.coutInt(clusters_segment,cluster_num);
  guideCenter(pts,clusters,guide_locs,clusters_segment,insts,insts.size(),cluster_num);
  cudaDeviceSynchronize();
  auto end2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed2 = end2 - end1;
  // std::cout << "guideCenter execution time: " << elapsed2.count() << " seconds" << std::endl;
  // levelProcess(guide_locs,pts,clusters,clusters_segment,insts,cluster_num,level,true);
  // clu.coutPointc(pts,insts.size());
  std::vector<Inst*> next_insts;
  GPUtoCPU(next_insts,guide_locs,cluster_num);
  check(next_insts);
  return next_insts;
}
}