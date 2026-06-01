#pragma once
#include "BoundSkewTree.h"
namespace gcts {
class Point3D {
 public:
    __host__ __device__ Point3D(double _x, double _y, int _z, int _count) : x(_x), y(_y),z(_z), count(_count) {}
    __host__ __device__ Point3D(double _x, double _y,int _z) : x(_x), y(_y),z(_z) {count=1;cap=0;instID=-1;}
    __host__ __device__ Point3D(){x=0,y=0,z=0,count=0;cap=0;instID=-1;}
    __host__ __device__ Point3D(double _x, double _y,int _z,int _count,double _cap) : x(_x), y(_y),z(_z),count(_count),cap(_cap) {}
    __host__ __device__ Point3D operator-(const Point3D& that) const { return Point3D(this->x - that.x, this->y - that.y,this->z,this->count,this->cap); }
    __host__ __device__ Point3D operator+(const Point3D& that) const { return Point3D(this->x + that.x, this->y + that.y,this->z,this->count,this->cap); }
    __host__ __device__ Point3D operator/(const int& that) const { return Point3D(this->x/that, this->y/that,this->z,this->count,this->cap); }
    __host__ __device__ Point3D operator*(const double& that) const { return Point3D(this->x*that, this->y*that,this->z,this->count,this->cap); }
     double x, y,cap;
     int z,count,clusterid,instID;
     __host__ __device__ bool operator<(const Point3D& other) const {
        return clusterid < other.clusterid; // 按key排序
    }
 };
class BoundSkewTree3D : public BoundSkewTree
{
 public:
  BoundSkewTree3D(int num,const std::optional<double>& skew_bound = std::nullopt);
  BoundSkewTree3D(Point3D* pts,Pointc* clusters,int* clusters_segment,int pinnum,int net_num);
  ~BoundSkewTree3D(){
    if(net_len!=nullptr)cudaFree(net_len);
    if(net_delay!=nullptr)cudaFree(net_delay);
  };
  void Point3Dto2D(Point3D* pts3d,Pointc* pts,int pinnum);
  void build3D(const double skew_bound, const int db_unit, const double unit_h_cap, const double unit_h_res,const double unit_v_cap, const double unit_v_res);
  void recursiveBottomUp3D();
  int *layers;
};
}


