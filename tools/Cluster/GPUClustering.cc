#include "MinCostFlow.hh"
#include "GPUClustering.h"
#include <random>
namespace gcts {
double setAps(std::vector<Inst*>& insts,Pointc* h_buffers,int cluster_num){
  std::vector<double> distances;
  for(int i=0; i<insts.size(); i++){
      for(int j=0; j<cluster_num; j++){
          double dist = std::fabs(insts[i]->get_location().x() - h_buffers[j].x) + 
                        std::fabs(insts[i]->get_location().y() - h_buffers[j].y);
          distances.push_back(dist);
      }
  }
std::sort(distances.begin(), distances.end());
double aps=30;
if(distances.size()>1000)aps = distances[distances.size() * 0.01]; 
return aps;
}
void GPUClustering::AllocateRun(std::vector<Inst*>& insts,Pointc* pts,Pointc* clusters,Pointc*buffers,int max_fanout,int cluster_num){
    gcts::MinCostFlow<Inst*> mcf;
    Pointc h_buffers[cluster_num];
    std::vector<Pointc>h_clusters;
    std::vector<Pointc>h_pts;
    std::vector<int>max_fanout_vector;
    std::vector<std::vector<int>>con_clusters(cluster_num);
    cudaMemcpy(h_buffers,buffers,cluster_num*sizeof(Pointc),cudaMemcpyDeviceToHost);
    double cost=0;
    int cnt=0,c=0;
    int f[insts.size()],fanout[cluster_num];
    for(int i=0;i<cluster_num;i++)fanout[i]=0;
    for(int i=0;i<insts.size();i++)f[i]=-1;
    // double aps=30.;
    double aps=setAps(insts,h_buffers,cluster_num);
    double Aps=aps;
    // std::cout<<aps<<'\n';
    for(int l=0;l< 20;l++){
      for(int i=0;i<insts.size();i++){
      if(f[i]!=-1)continue;
      for(int j=0;j<cluster_num;j++){ 
        if(f[i]==-1&&fanout[j]<max_fanout&&(std::fabs(insts[i]->get_location().x()-h_buffers[j].x)+std::fabs(insts[i]->get_location().y()-h_buffers[j].y))<Aps){
          f[i]=j,fanout[j]++;
          cost+=(std::fabs(insts[i]->get_location().x()-h_buffers[j].x)+std::fabs(insts[i]->get_location().y()-h_buffers[j].y));
          cnt++;
          break;
        }
      }
       if(f[i]!=-1)con_clusters[f[i]].push_back(i);
      }
      Aps+=aps;
      if(cnt==0)continue;
      // if(insts.size()/cnt<3)break;
      if((insts.size()-cnt)<3000)break;
    }
    for(int i=0;i<insts.size();i++)if(f[i]==-1)mcf.add_node(insts[i]->get_location().x(), insts[i]->get_location().y(), insts[i]);
    // for(int i=0;i<cluster_num;i++)std::cout<<fanout[i]<<' ';
    // std::cout<<'\n';
    // std::for_each(insts.begin(),insts.end(), [&mcf](Inst* inst) { mcf.add_node(inst->get_location().x(), inst->get_location().y(), inst); });
    std::for_each(&h_buffers[0],&h_buffers[cluster_num], [&mcf](const Pointc& buffer) { mcf.add_center(buffer.x, buffer.y); });
    int mx=0;
    for(int i=0;i<cluster_num;i++){
      max_fanout_vector.push_back(max_fanout-fanout[i]);
      // if(fanout[i]>max_fanout)std::cout<<"-------\n";
      mx=std::max(mx,fanout[i]);
    }
    // std::cout<<mx<<'\n';
    auto mcf_h_pts = mcf.run(max_fanout_vector);
    for(int j=0;j < mcf_h_pts.size();j++){
      auto h_pt = mcf_h_pts[j];
      double x=0,y=0,cap=0;
      for(int i=0;i<h_pt.size();i++){
        Pointc pt(h_pt[i]->get_location().x(),h_pt[i]->get_location().y(),1,h_pt[i]->cap);
        pt.instID=h_pt[i]->ID;
        x+=pt.x,y+=pt.y,cap+=pt.cap;
        h_pts.push_back(pt);
        cost+=(std::fabs(h_pt[i]->get_location().x()-h_buffers[j].x)+std::fabs(h_pt[i]->get_location().y()-h_buffers[j].y));
      }
      for(int i=0;i<con_clusters[j].size();i++){
        Pointc pt(insts[con_clusters[j][i]]->get_location().x(),insts[con_clusters[j][i]]->get_location().y(),1,insts[con_clusters[j][i]]->cap);
        pt.instID=insts[con_clusters[j][i]]->ID;
        x+=pt.x,y+=pt.y,cap+=pt.cap;
        h_pts.push_back(pt);
      }
      h_clusters.push_back(Pointc(x/(h_pt.size()+fanout[j]),y/(h_pt.size()+fanout[j]),h_pt.size()+fanout[j],cap));
    }
    // std::cout<<h_pts.size()<<' '<<h_clusters.size()<<'\n';
    cudaMemcpy(pts,h_pts.data(),h_pts.size()*sizeof(Pointc),cudaMemcpyHostToDevice);
    cudaMemcpy(clusters,h_clusters.data(),h_clusters.size()*sizeof(Pointc),cudaMemcpyHostToDevice);
    // std::cout<<"MinCostFlowRun\n";
    // initClusterSegment(clusters,clusters_segment,cluster_num);
}
double calcDistcpu(const Point& p1, const Point& p2){
    double dist = 0;
    dist = std::abs(p1.x() - p2.x()) + std::abs(p1.y() - p2.y());
    return dist;
}
std::vector<std::vector<Inst*>> GPUClustering::kMeans(const std::vector<Inst*>& insts, const size_t& k, const int& seed,
                                                          const size_t& max_iter){
  size_t num_instances = insts.size();
  std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
  std::uniform_int_distribution<> dis(0, num_instances - 1);

  std::vector<int> assignments(num_instances);
  std::vector<Point> centers(k);
  int cnt=0;
  centers[cnt++]=insts[dis(gen)]->get_location();
  while(cnt<k){
    std::vector<double> distances(num_instances, std::numeric_limits<double>::max());
    for (size_t i = 0; i < num_instances; i++) {
      double min_distance = std::numeric_limits<double>::max();
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = calcDistcpu(insts[i]->get_location(), centers[j]);
        min_distance = std::min(min_distance, distance);
      }
      distances[i] = min_distance * min_distance;  // square distance
    }
    std::discrete_distribution<> distribution(distances.begin(), distances.end());
    int selected_index = distribution(gen);
    auto select_loc = insts[selected_index]->get_location();
    centers[cnt++]=select_loc;
  }
  // std::cout<<"center\n";
  for (size_t iter = 0; iter < max_iter; ++iter) {
    std::vector<double> new_center_x(k, 0);
    std::vector<double> new_center_y(k, 0);
    std::vector<Point> new_centers(k);
    std::vector<int> center_counts(k, 0);
    for (size_t i = 0; i < num_instances; ++i) {
      double min_distance = std::numeric_limits<double>::max();
      int min_center_index = -1;
      for (size_t j = 0; j < k; ++j) {
        double distance = calcDistcpu(insts[i]->get_location(), centers[j]);
        if (distance < min_distance) {
          min_distance = distance;
          min_center_index = j;
        }
      }
      assignments[i] = min_center_index;
      new_center_x[min_center_index] += insts[i]->get_location().x();
      new_center_y[min_center_index] += insts[i]->get_location().y();
      center_counts[min_center_index]++;
    }
    for (size_t i = 0; i < k; ++i) {
      if (center_counts[i] > 0) {
        new_centers[i] = Point(new_center_x[i] / center_counts[i], new_center_y[i] / center_counts[i]);
      } else {
        // random choose a sink as new center
        auto loc = insts[dis(gen)]->get_location();
        new_centers[i] = loc;
      }
    }
    // if (new_centers == centers) {
    //   break;
    // }
    centers = new_centers;
  }
  std::vector<std::vector<Inst*>> best_clusters(k);
  for (size_t i = 0; i < num_instances; ++i) {
    int center_index = assignments[i];
    best_clusters[center_index].push_back(insts[i]);
  }
  // remove empty clusters
  best_clusters.erase(
      std::remove_if(best_clusters.begin(), best_clusters.end(), [](const std::vector<Inst*>& cluster) { return cluster.empty(); }),
      best_clusters.end());
  return best_clusters;
}
}