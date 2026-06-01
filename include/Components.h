#pragma once
#include <array>
#include <type_traits>
#include <vector>
#include <cuda_runtime.h>
#include <string> 

// static size_t kHead = 0;
// static size_t kTail = 1;
// static size_t kLeft = 0;
// static size_t kRight = 1;
// static size_t kMin = 0;
// static size_t kMax = 1;
// static size_t kX = 0;
// static size_t kY = 1;
// static size_t kH = 0;
// static size_t kV = 1;
constexpr static double kEpsilon = 1e-7;

#define FOR_EACH_SIDE(side) for (size_t side = 0; side < 2; ++side)
class Pt
{
 public:
  Pt() = default;
  Pt(const double& t_x, const double& t_y, const double& t_max, const double& t_min, const double& t_val)
      : x(t_x), y(t_y), max(t_max), min(t_min), val(t_val)
  {
  }
  __host__ __device__ Pt(const double& t_x, const double& t_y) : x(t_x), y(t_y), max(0), min(0), val(0) {}

  Pt operator+(const Pt& other) const { return Pt(x + other.x, y + other.y); }
  Pt operator-(const Pt& other) const { return Pt(x - other.x, y - other.y); }
  Pt operator*(const double& scale) const { return Pt(x * scale, y * scale); }
  Pt operator/(const double& scale) const { return Pt(x / scale, y / scale); }
  Pt operator+=(const Pt& other)
  {
    x += other.x;
    y += other.y;
    return *this;
  }
  Pt operator-=(const Pt& other)
  {
    x -= other.x;
    y -= other.y;
    return *this;
  }
  Pt operator*=(const double& scale)
  {
    x *= scale;
    y *= scale;
    return *this;
  }
  Pt operator/=(const double& scale)
  {
    x /= scale;
    y /= scale;
    return *this;
  }

  double x = 0;
  double y = 0;
  double max = 0;
  double min = 0;
  double val = 0;
};
// template <typename T>
// using Side = T[2];
// using Line = Pt[2];
template <typename T>
class Side {
  public:
    T data[2];
    __device__ __host__ Side() = default;
    __device__ __host__ Side(const T& a, const T& b) : data{a, b} {}
    __device__ __host__ T& operator[](int i) { return data[i]; }
    __device__ __host__ const T& operator[](int i) const { return data[i]; }
};
class Line {
  public:
    Line() = default;
    __host__ __device__ Line(Pt pt1,Pt pt2){pts[0]=pt1,pts[1]=pt2;}
    Pt pts[2];
};
class Region{
  public:
     Region(int num) : size(num), pts(nullptr) {}
     Region():size(0), pts(nullptr) {}
    //  int getsize(){return size;}
    //  void setsize(int num){size=num;}
    //  void setpts(Pt* _pts){pts=_pts;}
    Pt* pts;
    int size;
  private:
   
};
class Pts{
  public:
     Pts(int num) : size(num), pts(nullptr) {}
     Pts():size(0), pts(nullptr) {}
    //  int getsize(){return size;}
    //  void setsize(int num){size=num;}
    //  void setpts(Pt* _pts){pts=_pts;}
    Pt* pts;
    int size;
  private:
   
};
class Val{
  public:
  __device__ __host__ Val(){};
  double val;
  int id;
};
class Area{
 public:
  __device__ __host__ Area(){left=-1,right=-1,p=-1;};
  Area(const int & id) { type = false;};//mr_lines初始化为空
  __device__ __host__ Area(const double& _x, const double& _y)  : x(_x), y(_y){left=-1,right=-1,p=-1;};
  __device__ __host__ Area(const double& _x, const double& _y,const double& _cap)  : x(_x), y(_y),cap(_cap){left=-1,right=-1,p=-1;};
  __host__ __device__ bool operator<(const Area& other) const {
        return val < other.val; // 按key排序
    }
  double x;
  double y;
  bool type;
  int left = -1;
  int right =-1;
  int size = -1;
  int p=-1;
  int l=-1;
  int r=-1;
  double val;
  Region mr;
  Region convex_hull;
  Line *mr_lines;
  Side<Line> lines;
  double radius = 0;
  Area* get_left=nullptr;
  Area* get_right=nullptr;
  double cap;
  Pt location;
  Side<double> _edge_len = {0, 0};
  double snake;
  int id;
  int ptsid;
  // std::string name;
};

class Interval
{
 public:
  Interval() = default;
  Interval(const double& val) : _low(val), _high(val) {}
  __device__ __host__ Interval(const double& low, const double& high) : _low(low), _high(high) {}
  Interval(const Interval& other) : _low(other._low), _high(other._high) {}

  const double& low() const { return _low; }
  const double& high() const { return _high; }

  __device__ __host__ bool is_empty() const { return _low > _high; }
  bool is_point() const { return _low == _high; }

  void enclose(const double& val)
  {
    if (is_empty()) {
      _low = val;
      _high = val;
    } else {
      _low = std::min(_low, val);
      _high = std::max(_high, val);
    }
  }
  void enclose(const Interval& other)
  {
    if (!other.is_empty()) {
      enclose(other.low());
      enclose(other.high());
    }
  }

  bool isEnclosed(const double& val) const { return _low <= val && val <= _high; }
  bool isEnclosed(const Interval& other) const { return _low <= other.low() && other.high() <= _high; }

  double width() const { return is_empty() ? 0 : _high - _low; }

 private:
  double _low = 1;
  double _high = 0;
};
class Trr
{
 public:
  Trr() = default;
  Trr(const double& x_low, const double& x_high, const double& y_low, const double& y_high)
      : _x_low(x_low), _x_high(x_high), _y_low(y_low), _y_high(y_high)
  {
  }
  __host__ __device__ Trr(const Pt& point, const double& radius) { makeDiamond(point, radius); }

  void init()
  {
    _x_low = _y_low = 1;
    _x_high = _y_high = 0;
  }
  const double& x_low() const { return _x_low; }
  const double& x_high() const { return _x_high; }
  const double& y_low() const { return _y_low; }
  const double& y_high() const { return _y_high; }
  void x_low(const double& val) { _x_low = val; }
  void x_high(const double& val) { _x_high = val; }
  void y_low(const double& val) { _y_low = val; }
  void y_high(const double& val) { _y_high = val; }

  bool is_empty() const
  {
    auto x_interval = Interval(_x_low, _x_high);
    auto y_interval = Interval(_y_low, _y_high);
    return x_interval.is_empty() || y_interval.is_empty();
  }
  __host__ __device__ void makeDiamond(const Pt& point, const double& radius)
  {
    auto val = point.x - point.y;
    _x_low = val - radius;
    _x_high = val + radius;
    val = point.x + point.y;
    _y_low = val - radius;
    _y_high = val + radius;
  }
  void enclose(const Trr& other)
  {
    if (is_empty()) {
      _x_low = other._x_low;
      _x_high = other._x_high;
      _y_low = other._y_low;
      _y_high = other._y_high;
    } else {
      _x_low = std::min(_x_low, other._x_low);
      _x_high = std::max(_x_high, other._x_high);
      _y_low = std::min(_y_low, other._y_low);
      _y_high = std::max(_y_high, other._y_high);
    }
  }

  double width(const size_t& side) const
  {
    if (side == 0) {
      return _x_high - _x_low;
    } else {
      return _y_high - _y_low;
    }
  }

  double diameter() const { return std::max(width(0), width(1)); }

  Trr intersect(const Trr& trr1, const Trr& trr2)
  {
    auto x_low = std::max(trr1._x_low, trr2._x_low);
    auto x_high = std::min(trr1._x_high, trr2._x_high);
    auto y_low = std::max(trr1._y_low, trr2._y_low);
    auto y_high = std::min(trr1._y_high, trr2._y_high);
    auto trr = Trr(x_low, x_high, y_low, y_high);
    // trr.check();
    return trr;
  }
  void check()
  {
    // correction();
    // LOG_FATAL_IF(is_empty()) << "TRR is empty, which x_low: " << _x_low << ", x_high: " << _x_high << ", y_low: " << _y_low
    //                          << ", y_high: " << _y_high;
  }

  // void correction()
  // {
  //   auto temp_low = _x_low;
  //   auto temp_high = _x_high;
  //   if (Equal(temp_low, temp_high)) {
  //     _x_low = _x_high = (temp_low + temp_high) / 2;
  //   }
  //   temp_low = _y_low;
  //   temp_high = _y_high;
  //   if (Equal(temp_low, temp_high)) {
  //     _y_low = _y_high = (temp_low + temp_high) / 2;
  //   }
  // }
  double _x_low = 1;
  double _x_high = 0;
  double _y_low = 1;
  double _y_high = 0;
};

