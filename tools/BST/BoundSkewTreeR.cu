#include <stack>
#include <cub/cub.cuh>
// #include "GeomCalc.cuh"
#include "BoundSkewTree.h"
#include <chrono>
#define CUDA_CHECK(call) \
{ \
    cudaError_t err = (call); \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
namespace gcts {
__device__ size_t kHead;
__device__ size_t kTail;
__device__ size_t kLeft;
__device__ size_t kRight;
__device__ size_t kMin;
__device__ size_t kMax;
__device__ size_t kX;
__device__ size_t kY;
__device__ size_t kH;
__device__ size_t kV;
__device__ Side<double> _K;
__device__ RCPattern _pattern;
__device__ double DOUBLE_MAX;
__device__ double DOUBLE_MIN;
__device__  int _db_unit;
__device__  double _unit_h_cap;
__device__  double _unit_h_res;
__device__  double _unit_v_cap;
__device__  double _unit_v_res;
__device__  double _skew_bound;
 __device__  double kEpsilon;
__device__ double BoundSkewTree::crossProductR(const Pt& p1, const Pt& p2, const Pt& p3){
  return (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y);
}
__device__ double BoundSkewTree::distanceR(const Pt& p1, const Pt& p2){
  return fabs(p1.x - p2.x) + fabs(p1.y - p2.y);
}
__device__ bool BoundSkewTree::Equal(const double& a, const double& b){
  return fabs(a - b) < 1e-7;
}
__device__ void BoundSkewTree::checkMs(Trr& ms){
  auto x_low = ms._x_low;
  auto x_high = ms._x_high;
  if (Equal(x_low, x_high)) {
    auto avg = (x_low + x_high) / 2;
    ms._x_low=avg;
    ms._x_high=avg;
  }
  auto y_low = ms._y_low;
  auto y_high = ms._y_high;
  if (Equal(y_low, y_high)) {
    auto avg = (y_low + y_high) / 2;
    ms._y_low=avg;
    ms._y_high=avg;
  }
  // LOG_FATAL_IF(ms.x_low() > ms.x_high() || ms.y_low() > ms.y_high()) << "ms is not valid";
}
__device__ bool BoundSkewTree::onLine(Pt& p, const Line& l){
  auto len = distanceR(l.pts[kHead], l.pts[kTail]);
  auto len_to_head = distanceR(p, l.pts[kHead]);
  auto len_to_tail = distanceR(p, l.pts[kTail]);
  if (fabs(len_to_head + len_to_tail - len) < 2 * kEpsilon) {
    if (Equal(len_to_head, 0)) {
      p = l.pts[kHead];
      return true;
    } else if (Equal(len_to_tail, 0)) {
      p = l.pts[kTail];
      return true;
    } else {
      auto delta_x = fabs(l.pts[kTail].x - l.pts[kHead].x);
      auto delta_y = fabs(l.pts[kTail].y - l.pts[kHead].y);
      if (delta_y > delta_x) {
        auto temp_x = (l.pts[kTail].x - l.pts[kHead].x) * (p.y - l.pts[kHead].y) / (l.pts[kTail].y - l.pts[kHead].y) + l.pts[kHead].x;
        if (Equal(temp_x, p.x)) {
          p.x = temp_x;
          return true;
        }
      } else {
        auto temp_y = (l.pts[kTail].y - l.pts[kHead].y) * (p.x - l.pts[kHead].x) / (l.pts[kTail].x - l.pts[kHead].x) + l.pts[kHead].y;
        if (Equal(temp_y, p.y)) {
          p.y = temp_y;
          return true;
        }
      }
    }
  }
  return false;
}
__device__ bool BoundSkewTree::isSegmentTrr(const Trr& trr)
{
  return Equal(trr._x_low, trr._x_high) || Equal(trr._y_low, trr._y_high);
}
__device__ void BoundSkewTree::msToLine(Trr& ms, Pt& p1, Pt& p2){
  checkMs(ms);
  p1.x = (ms._y_low + ms._x_high) / 2;
  p1.y = (ms._y_low - ms._x_high) / 2;
  p2.x = (ms._y_high + ms._x_low) / 2;
  p2.y = (ms._y_high - ms._x_low) / 2;
  // LOG_FATAL_IF(p1.y > p2.y) << "p2.y should be larger than p1.y";
}
__device__ void BoundSkewTree::msToLine(Trr& ms, Line& l){
  msToLine(ms, l.pts[kHead], l.pts[kTail]);
}
__device__ void BoundSkewTree::trrCore(const Trr& trr, Trr& core){
  if (trr._x_high - trr._x_low < trr._y_high - trr._y_low) {
    core._y_low = trr._y_low;
    core._y_high = trr._y_high;
    auto x = (trr._x_low + trr._x_high) / 2;
    core._x_low=x;
    core._x_high=x;
  } else {
    core._x_low=trr._x_low;
    core._x_high=trr._x_high;
    auto y = (trr._y_low + trr._y_high) / 2;
    core._y_low=y;
    core._y_high=y;
  }
}
__device__ bool BoundSkewTree::isSame(const Pt& p1, const Pt& p2){
  auto dist = distanceR(p1, p2);
  return Equal(dist, 0);
}
__device__ bool BoundSkewTree::boundBoxOverlap(const double& x1, const double& y1, const double& x2, const double& y2, const double& x3, const double& y3,
                               const double& x4, const double& y4, const double& epsilon){
  if (fmin(x1, x2) - fmax(x3, x4) >= epsilon) {
    return false;
  }
  if (fmin(x3, x4) - fmax(x1, x2) >= epsilon) {
    return false;
  }
  if (fmin(y1, y2) - fmax(y3, y4) >= epsilon) {
    return false;
  }
  if (fmin(y3, y4) - fmax(y1, y2) >= epsilon) {
    return false;
  }
  return true;
}
__device__ bool BoundSkewTree::boundBoxOverlap(const Line& l1, const Line& l2, const double& epsilon){
  return boundBoxOverlap(l1.pts[kHead].x, l1.pts[kHead].y, l1.pts[kTail].x, l1.pts[kTail].y, l2.pts[kHead].x, l2.pts[kHead].y, l2.pts[kTail].x, l2.pts[kTail].y, epsilon);
}
__device__ bool BoundSkewTree::inBoundBox(const Pt& p, const Line& l){
  if (p.x <= fmax(l.pts[kHead].x, l.pts[kTail].x) + kEpsilon && p.x >= fmin(l.pts[kHead].x, l.pts[kTail].x) - kEpsilon
      && p.y <= fmax(l.pts[kHead].y, l.pts[kTail].y) + kEpsilon && p.y >= fmin(l.pts[kHead].y, l.pts[kTail].y) - kEpsilon) {
    return true;
  }
  return false;
}
__device__ void sortPtsByVal(Pts& pts){
  if (pts.size==0) {
    return;
  }
  // std::ranges::sort(pts, [](const Pt& p1, const Pt& p2) { return p1.val < p2.val; });
  for (int i = 0; i < pts.size - 1; ++i) {
        for (int j = 0; j < pts.size - i - 1; ++j) {
            if (pts.pts[j].val > pts.pts[j+1].val) {
                // Swap elements
                Pt temp = pts.pts[j];
                pts.pts[j] = pts.pts[j+1];
                pts.pts[j+1] = temp;
            }
        }
    }
}
__device__ void sortPtsByVal(Pt* pts,int size){
  if (size==0) {
    return;
  }
  // std::ranges::sort(pts, [](const Pt& p1, const Pt& p2) { return p1.val < p2.val; });
  for (int i = 0; i < size - 1; ++i) {
        for (int j = 0; j < size - i - 1; ++j) {
            if (pts[j].val > pts[j+1].val) {
                // Swap elements
                Pt temp = pts[j];
                pts[j] = pts[j+1];
                pts[j+1] = temp;
            }
        }
    }
}
__device__ void sortPtsByFront(Pts& pts) {
  // sort by dist to first point
  // std::ranges::for_each(pts, [&pts](Pt& p) { p.val = distance(p, pts.front()); });
  for(int i=0;i<pts.size;i++){pts.pts[i].val=BoundSkewTree::distanceR(pts.pts[i], pts.pts[0]);}
  sortPtsByVal(pts);
}
__device__ void sortPtsByFront(Pt* pts,int size) {
  // sort by dist to first point
  // std::ranges::for_each(pts, [&pts](Pt& p) { p.val = distance(p, pts.front()); });
  for(int i=0;i<size;i++){pts[i].val=BoundSkewTree::distanceR(pts[i], pts[0]);}
  sortPtsByVal(pts,size);
}
__device__ void uniquePtsLoc(Pts& pts){
  if (pts.size < 2) {
    return;
  }
  Pt unique_pts[32];
  int cnt=0;
  unique_pts[cnt++] = pts.pts[0];
  // std::ranges::for_each(pts, [&unique_pts](const Pt& p) {
  for(int i=0;i<pts.size;i++){
    if (!BoundSkewTree::isSame(pts.pts[i], unique_pts[cnt-1])) {
      unique_pts[cnt++]=pts.pts[i];
    }
  }
  if (cnt > 1 && BoundSkewTree::isSame(unique_pts[0], unique_pts[cnt-1])) {
    cnt--;
  }
  for(int i=0;i<cnt;i++)pts.pts[i]=unique_pts[i];
  pts.size=cnt;
}
__device__ void uniquePtsLoc(Region& pts){
  if (pts.size < 2) {
    return;
  }
  Pt unique_pts[32];
  int cnt=0;
  unique_pts[cnt++] = pts.pts[0];
  // std::ranges::for_each(pts, [&unique_pts](const Pt& p) {
  for(int i=0;i<pts.size;i++){
    if (!BoundSkewTree::isSame(pts.pts[i], unique_pts[cnt-1])) {
      unique_pts[cnt++]=pts.pts[i];
    }
  }
  if (cnt > 1 && BoundSkewTree::isSame(unique_pts[0], unique_pts[cnt-1])) {
    cnt--;
  }
  for(int i=0;i<cnt;i++)pts.pts[i]=unique_pts[i];
  pts.size=cnt;
}
__device__ void uniquePtsLoc(Pt* pts,int &size){//size有数值
  if (size < 2) {
    return;
  }
  Pt unique_pts[32];
  int cnt=0;
  unique_pts[cnt++] = pts[0];
  // std::ranges::for_each(pts, [&unique_pts](const Pt& p) {
  for(int i=0;i<size;i++){
    if (!BoundSkewTree::isSame(pts[i], unique_pts[cnt-1])) {
      unique_pts[cnt++]=pts[i];
    }
  }
  if (cnt > 1 && BoundSkewTree::isSame(unique_pts[0], unique_pts[cnt-1])) {
    cnt--;
  }
  for(int i=0;i<cnt;i++)pts[i]=unique_pts[i];
  size=cnt;
}
__device__ void BoundSkewTree::trrToPt(const Trr& trr, Pt& pt)
{
  pt.x = (trr._y_low + trr._x_high) / 2;
  pt.y = (trr._y_low - trr._x_high) / 2;
}
__device__ void BoundSkewTree::trrToRegion(Trr& trr, Pt* region,int &cnt){
  auto x = trr._x_high - trr._x_low;
  auto y = trr._y_high - trr._y_low;
  if (Equal(x, 0) && Equal(y, 0)) {
    Pt pt;
    trrToPt(trr, pt);
    // region.push_back(pt);
    region[cnt++]=pt;
    // printf("cnt:%d",cnt);
    return;
  } else if (Equal(x, 0) || Equal(y, 0)) {
    Pt head, tail;
    msToLine(trr, head, tail);
    region[cnt++]=head;
    region[cnt++]=tail;
    // region.push_back(head);
    // region.push_back(tail);
    // printf("cnt:%d",cnt);
    return;
  }

  // Trr ms(trr._x_high, trr._x_high, trr._y_low, trr._y_high);
  Trr ms;
  ms._x_low=trr._x_high, ms._x_high= trr._x_high,ms._y_low=trr._y_low, ms._y_high=trr._y_high;
  Pt head, tail;

  msToLine(ms, head, tail);
  // region.push_back(head);
  // region.push_back(tail);
  region[cnt++]=head;
  region[cnt++]=tail;

  ms._x_low=trr._x_low;
  ms._x_high=trr._x_low;

  msToLine(ms, head, tail);
  // region.push_back(head);
  // region.push_back(tail);
  region[cnt++]=head;
  region[cnt++]=tail;
  // printf("cnt:%d",cnt);
}
__device__ void sortPtsByValDec(Pt* pts,int size){
  if (size==0) {
    return;
  }
  // std::ranges::sort(pts, [](const Pt& p1, const Pt& p2) { return p1.val > p2.val; });
  for (int i = 0; i < size - 1; ++i) {
        for (int j = 0; j < size - i - 1; ++j) {
            if (pts[j].val < pts[j+1].val) {  // 降序排序
                // 交换元素
                Pt temp = pts[j];
                pts[j] = pts[j+1];
                pts[j+1] = temp;
            }
        }
    }
}

__device__ IntersectType BoundSkewTree::lineIntersect(Pt& p, Line& l1, Line& l2){
  // LOG_FATAL_IF(Equal(distance(l1[kHead], l1[kTail]), 0) || Equal(distance(l2[kHead], l2[kTail]), 0)) << "line length is zero";
  if (!boundBoxOverlap(l1, l2,kEpsilon)) {
    return IntersectType::kNone;
  }
  if (distanceR(l1.pts[kHead], l2.pts[kHead]) + distanceR(l1.pts[kTail], l2.pts[kTail]) < kEpsilon) {
    p = l1.pts[kHead];
    return IntersectType::kSame;
  }
  size_t count = 0;
  FOR_EACH_SIDE(side)
  {
    if (onLine(l1.pts[side], l2)) {
      p = l1.pts[side];
      ++count;
    }
  }
  FOR_EACH_SIDE(side)
  {
    if (onLine(l2.pts[side], l1)) {
      p = l2.pts[side];
      ++count;
    }
  }
  FOR_EACH_SIDE(left_side)
  {
    FOR_EACH_SIDE(right_side)
    {
      if (distanceR(l1.pts[left_side], l2.pts[right_side]) < kEpsilon) {
        --count;
      }
    }
  }

  if (count >= 2) {
    return IntersectType::kOverlap;
  }

  auto l1_x_is_equal = Equal(l1.pts[kHead].x, l1.pts[kTail].x);
  auto l2_x_is_equal = Equal(l2.pts[kHead].x, l2.pts[kTail].x);
  if (l1_x_is_equal && l2_x_is_equal) {
    // parallel vertical lines
  } else if (l1_x_is_equal && !l2_x_is_equal) {
    p.x = l1.pts[kHead].x;
    p.y = (l2.pts[kHead].y - l2.pts[kTail].y) * (l1.pts[kHead].x - l2.pts[kHead].x) / (l2.pts[kHead].x - l2.pts[kTail].x) + l2.pts[kHead].y;
  } else if (!l1_x_is_equal && l2_x_is_equal) {
    p.x = l2.pts[kHead].x;
    p.y = (l1.pts[kTail].y - l1.pts[kHead].y) * (l2.pts[kHead].x - l1.pts[kHead].x) / (l1.pts[kTail].x - l1.pts[kHead].x) + l1.pts[kHead].y;
  } else if (!l1_x_is_equal && !l2_x_is_equal) {
    auto l1_k = (l1.pts[kTail].y - l1.pts[kHead].y) / (l1.pts[kTail].x - l1.pts[kHead].x);
    auto l2_k = (l2.pts[kTail].y - l2.pts[kHead].y) / (l2.pts[kTail].x - l2.pts[kHead].x);
    if (Equal(l1_k, l2_k)) {
      // parallel lines
      return IntersectType::kNone;
    } else {
      p.x = (l2.pts[kHead].y - l1.pts[kHead].y + l1.pts[kHead].x * l1_k - l2.pts[kHead].x * l2_k) / (l1_k - l2_k);
      p.y = (l1.pts[kHead].y + l2.pts[kHead].y + l1_k * (p.x - l1.pts[kHead].x) + l2_k * (p.x - l2.pts[kHead].x)) / 2.0;
    }
  }
  if (inBoundBox(p, l1) && inBoundBox(p, l2)) {
    return IntersectType::kCrossing;
  }
  return IntersectType::kNone;
}
__device__ void BoundSkewTree::lineToMs(Trr& ms, const Pt& p1, const Pt& p2){
  if (p1.y <= p2.y) {
    ms._x_low=p2.x - p2.y;
    ms._x_high=p1.x - p1.y;
    ms._y_low=p1.x + p1.y;
    ms._y_high=p2.x + p2.y;
  } else {
    ms._x_low=p1.x - p1.y;
    ms._x_high=p2.x - p2.y;
    ms._y_low=p2.x + p2.y;
    ms._y_high=p1.x + p1.y;
  }
  checkMs(ms);
}
__device__ void BoundSkewTree::lineToMs(Trr& ms, const Line& l){
  lineToMs(ms, l.pts[kHead], l.pts[kTail]);
}
__device__ double BoundSkewTree::msDistance(Trr& ms1, Trr& ms2){
  checkMs(ms1);
  checkMs(ms2);
  // Trr<T> is in a Linfinity metric space
  auto x1_low = ms1._x_low;
  auto x1_high = ms1._x_high;
  auto y1_low = ms1._y_low;
  auto y1_high = ms1._y_high;
  auto x2_low = ms2._x_low;
  auto x2_high = ms2._x_high;
  auto y2_low = ms2._y_low;
  auto y2_high = ms2._y_high;
  /*
  (1) Not intersect between x-coords and y-coords
  (2) Not intersect between x-coords, but intersect between y-coords
  (3) Not intersect between y-coords, but intersect between x-coords
  (4) Intersect between x-coords and y-coords (distance = 0)
  */
  auto x_is_intersect = (x1_low <= x2_high) && (x2_low <= x1_high);
  auto y_is_intersect = (y1_low <= y2_high) && (y2_low <= y1_high);
  if (!x_is_intersect && !y_is_intersect) {
    // (1)
    auto low1_to_low2 = fmax(fabs(x1_low - x2_low), fabs(y1_low - y2_low));
    auto low1_to_high2 = fmax(fabs(x1_low - x2_high), fabs(y1_low - y2_high));
    auto high1_to_low2 = fmax(fabs(x1_high - x2_low), fabs(y1_high - y2_low));
    auto high1_to_high2 = fmax(fabs(x1_high - x2_high), fabs(y1_high - y2_high));
    return fmin(fmin(low1_to_low2, low1_to_high2), fmin(high1_to_low2, high1_to_high2));
  } else if (!x_is_intersect) {
    // (2)
    auto low1_to_high2 = x1_low - x2_high;
    auto low2_to_high1 = x2_low - x1_high;
    return fmax(low1_to_high2, low2_to_high1);
  } else if (!y_is_intersect) {
    // (3)
    auto low1_to_high2 = y1_low - y2_high;
    auto low2_to_high1 = y2_low - y1_high;
    return fmax(low1_to_high2, low2_to_high1);
  } else {
    // (4)
    return 0;
  }
}
__device__ void BoundSkewTree::makeIntersect(Trr& ms1, Trr& ms2, Trr& intersect){
  intersect._x_low=fmax(ms1._x_low, ms2._x_low);
  intersect._x_high=fmin(ms1._x_high, ms2._x_high);
  intersect._y_low=fmax(ms1._y_low, ms2._y_low);
  intersect._y_high=fmin(ms1._y_high, ms2._y_high);
  checkMs(intersect);
}
__device__ void BoundSkewTree::coreMidPoint(Trr& ms, Pt& mid){
  auto x = (ms._x_low + ms._x_high) / 2;
  auto y = (ms._y_low + ms._y_high) / 2;
  mid.x = (x + y) / 2;
  mid.y = (y - x) / 2;
}
__device__ double BoundSkewTree::ptToLineDistManhattan(Pt& p, const Line& l, Pt& closest){
  // manhattan arc
  Trr ms_p = Trr(p, 0);
  Trr ms_l;
  lineToMs(ms_l, l);
  auto dist = msDistance(ms_p, ms_l);
  Trr new_ms_p;
  new_ms_p.makeDiamond(p, dist);
  Trr intersect_ms;
  makeIntersect(new_ms_p, ms_l, intersect_ms);
  coreMidPoint(intersect_ms, closest);
  return dist;
}
__device__ double BoundSkewTree::ptToLineDistNotManhattan(Pt& p, const Line& l, Pt& closest){
  Pt candidate[4];
  int cnt=0;
  if (!Equal(l.pts[kHead].x, l.pts[kTail].x) && (p.x - l.pts[kHead].x) * (p.x - l.pts[kTail].x) <= 0) {
    Pt pt;
    pt.x = p.x;
    pt.y = (l.pts[kTail].x - p.x) * (l.pts[kHead].y - l.pts[kTail].y) / (l.pts[kTail].x - l.pts[kHead].x) + l.pts[kTail].y;
    candidate[cnt++]=pt;
  }
  if (!Equal(l.pts[kHead].y, l.pts[kTail].y) && (p.y - l.pts[kHead].y) * (p.y - l.pts[kTail].y) <= 0) {
    Pt pt;
    pt.x = (l.pts[kTail].y - p.y) * (l.pts[kHead].x - l.pts[kTail].x) / (l.pts[kTail].y - l.pts[kHead].y) + l.pts[kTail].x;
    pt.y = p.y;
    candidate[cnt++]=pt;
  }
  if (cnt < 2) {
    candidate[cnt++]=l.pts[kHead];
    candidate[cnt++]=l.pts[kTail];
  }
  double min_dist = DOUBLE_MAX;
  // std::ranges::for_each(candidate, [&p, &min_dist, &closest](auto& pt) {
  for(int i=0;i<cnt;i++){
    auto dist = distanceR(p, candidate[i]);
    if (dist < min_dist) {
      min_dist = dist;
      closest = candidate[i];
    }
  }
  return min_dist;
}
__device__ double BoundSkewTree::ptToLineDist(Pt& p, const Line& l, Pt& closest){
  auto min_dist = DOUBLE_MAX;
  auto delta_x = fabs(l.pts[kHead].x - l.pts[kTail].x);
  auto delta_y = fabs(l.pts[kHead].y - l.pts[kTail].y);
  if (isSame(l.pts[kHead], l.pts[kTail])) {
    closest = l.pts[kHead];
    min_dist = distanceR(p, closest);
  } else if (onLine(p, l)) {
    closest = p;
    min_dist = 0;
  } else if (Equal(delta_x, delta_y)) {
    // manhattan arc
    min_dist = ptToLineDistManhattan(p, l, closest);
  } else {
    // not manhattan arc
    min_dist = ptToLineDistNotManhattan(p, l, closest);
  }
  // LOG_FATAL_IF(onLine(closest, l) == false) << "closest point is not on line";
  return min_dist;
}
__device__ double BoundSkewTree::lineDist(Line& l1, Line& l2, PtPair& closest){
  double dist, min_dist = DOUBLE_MAX;
  Pt intersect;
  Side<Side<Pt>> pt;
  pt[kLeft][0] = l1.pts[0];  
  pt[kLeft][1] = l1.pts[1];
  // pt[kLeft] = l1;
  // pt[kRight] = l2;
  pt[kRight][0] = l2.pts[0];  
  pt[kRight][1] = l2.pts[1];
  size_t n1 = 2;
  size_t n2 = 2;
  if (isSame(l1.pts[kHead], l1.pts[kTail])) {
    n1 = 1;
  }
  if (isSame(l2.pts[kHead], l2.pts[kTail])) {
    n2 = 1;
  }
  if (n1 == 2 && n2 == 2) {
    if (lineIntersect(intersect, l1, l2) != IntersectType::kNone) {
      closest[kLeft] = closest[kRight] = intersect;
      return 0;
    }
  }
  for (size_t i = 0; i < n1; ++i) {
    size_t k = (i + 1) % 2;
    for (size_t j = 0; j < n2; ++j) {
      // dist = ptToLineDist(pt[i][j], pt[k], intersect);
      Line l;
      l.pts[0]=pt[k][0],l.pts[1]=pt[k][1];
      dist = BoundSkewTree::ptToLineDist(pt[i][j],l, intersect); //newbug
      if (dist < min_dist) {
        min_dist = dist;
        closest[i] = pt[i][j];
        closest[k] = intersect;
      }
    }
  }
  return min_dist;
}
__device__ double calcDelayIncrease(const double& x, const double& y, const double& cap, const RCPattern& pattern){
  double delay = 0;
  switch (pattern) {
    case RCPattern::kHV:
      delay = _unit_h_res * x * (_unit_h_cap * x / 2 + cap) + _unit_v_res * y * (_unit_v_cap * y / 2 + cap + x * _unit_h_cap);
      break;
    case RCPattern::kVH:
      delay = _unit_v_res * y * (_unit_v_cap * y / 2 + cap) + _unit_h_res * x * (_unit_h_cap * x / 2 + cap + y * _unit_v_cap);
      break;
    case RCPattern::kSingle:
      delay = _unit_h_res * (x + y) * (_unit_h_cap * (x + y) / 2 + cap);
      break;
    default:
      // LOG_FATAL << "unknown pattern";
      break;
  }
  return delay;
}
__device__ double ptDelayIncrease(Pt& p1, Pt& p2, const double& cap, const RCPattern& pattern){
  auto delay = calcDelayIncrease(fabs(p1.x - p2.x), fabs(p1.y - p2.y), cap, pattern);
  // LOG_FATAL_IF(delay < 0) << "point increase delay is negative";
  return delay;
}
__device__ void setJrLine(const size_t& side, const Line& line, Side<Pts>& _join_region){
  _join_region[side].pts[kHead] = line.pts[kHead];
  _join_region[side].pts[kTail] = line.pts[kTail];
  _join_region[side].size=max(_join_region[side].size,2);
}
__device__ Line getJsLine(const Side<Pts>& _join_segment,const size_t& side){
  auto js = _join_segment[side];
  // printf("getJsLine: %f %f\n",js.pts[kHead].x,js.pts[kHead].y);
  return Line(js.pts[kHead], js.pts[kTail]);
}
__device__ void setJsLine(const size_t& side, const Line& line,Side<Pts>& _join_segment){
  _join_segment[side].pts[kHead] = line.pts[kHead];
  _join_segment[side].pts[kTail] = line.pts[kTail];
  _join_segment[side].size = max(_join_segment[side].size, 2);
}
__device__ void buildTrr(const Trr& ms, const double& r, Trr& build_trr){
  build_trr._x_low=ms._x_low - r;
  build_trr._x_high=ms._x_high + r;
  build_trr._y_low=ms._y_low - r;
  build_trr._y_high=ms._y_high + r;
}
__device__ LineType BoundSkewTree::lineType(const Pt& p1, const Pt& p2)
{
  auto d_x = fabs(p1.x - p2.x);
  auto d_y = fabs(p1.y - p2.y);
  if (Equal(d_x, d_y)) {
    return LineType::kManhattan;
  } else if (Equal(d_x, 0)) {
    return LineType::kVertical;
  } else if (Equal(d_y, 0)) {
    return LineType::kHorizontal;
  } else if (d_x > d_y) {
    return LineType::kFlat;
  } else {
    return LineType::kTilt;
  }
}
__device__ LineType BoundSkewTree::lineType(const Line& l){
  return lineType(l.pts[kHead], l.pts[kTail]);
}

__device__ void BoundSkewTree::calcCoord(Pt& p, const Line& l, const double& shift){
  auto line_type = lineType(l);
  double d0 = 0;
  double d1 = 0;
  if (line_type == LineType::kHorizontal || line_type == LineType::kFlat) {
    d0 = fabs(l.pts[kHead].x - shift);
    d1 = fabs(l.pts[kTail].x - shift);
  } else {
    d0 = fabs(l.pts[kHead].y - shift);
    d1 = fabs(l.pts[kTail].y - shift);
  }
  p.x = (l.pts[kHead].x * d1 + l.pts[kTail].x * d0) / (d0 + d1);
  p.y = (l.pts[kHead].y * d1 + l.pts[kTail].y * d0) / (d0 + d1);
}
__device__ bool BoundSkewTree::isParallel(const Line& l1, const Line& l2){
  if (isSame(l1.pts[kHead], l1.pts[kTail]) || isSame(l2.pts[kHead], l2.pts[kTail])) {
    return false;
  }
  auto delta_x1 = fabs(l1.pts[kHead].x - l1.pts[kTail].x);
  auto delta_y1 = fabs(l1.pts[kHead].y - l1.pts[kTail].y);
  auto delta_x2 = fabs(l2.pts[kHead].x - l2.pts[kTail].x);
  auto delta_y2 = fabs(l2.pts[kHead].y - l2.pts[kTail].y);
  if (Equal(delta_x1, 0) && Equal(delta_x2, 0)) {
    return true;
  } else if (!Equal(delta_x1, 0) && !Equal(delta_x2, 0) && Equal(delta_y1 / delta_x1, delta_y2 / delta_x2)) {
    return true;
  }
  return false;
}
__device__ void updateJS(Area* cur, Line& left, Line& right, PtPair& closest, Side<Pts>& _join_segment,Side<Trr>&_ms){//bug
  auto left_type = BoundSkewTree::lineType(left);
  auto right_type = BoundSkewTree::lineType(right);
  auto left_is_manhattan = left_type == LineType::kManhattan;
  auto right_is_manhattan = right_type == LineType::kManhattan;
  Trr left_ms, right_ms;
  if (left_is_manhattan) {
    BoundSkewTree::lineToMs(left_ms, left);
  }
  if (right_is_manhattan) {
    BoundSkewTree::lineToMs(right_ms, right);
  }
  if (!left_is_manhattan && right_is_manhattan) {
    left_ms.makeDiamond(closest[kLeft], 0);
  }
  if (left_is_manhattan && !right_is_manhattan) {
    right_ms.makeDiamond(closest[kRight], 0);
  }
  setJsLine(kLeft, {closest[kLeft], closest[kLeft]},_join_segment);
  setJsLine(kRight, {closest[kRight], closest[kRight]},_join_segment);
  if (left_is_manhattan || right_is_manhattan) {
    auto dist = BoundSkewTree::msDistance(left_ms, right_ms);
    // LOG_FATAL_IF(std::abs(dist - cur->get_radius()) > kEpsilon) << "ms distance is not equal to radius";
    cur->radius=dist;
    _ms[kLeft] = left_ms;
    _ms[kRight] = right_ms;
    Trr left_bound, right_bound, left_intersect, right_intersect;
    buildTrr(left_ms, dist, left_bound);
    buildTrr(right_ms, dist, right_bound);
    BoundSkewTree::makeIntersect(right_bound, left_ms, left_intersect);
    BoundSkewTree::makeIntersect(left_bound, right_ms, right_intersect);
    BoundSkewTree::msToLine(left_intersect, _join_segment[kLeft].pts[kHead], _join_segment[kLeft].pts[kTail]);
    BoundSkewTree::msToLine(right_intersect, _join_segment[kRight].pts[kHead], _join_segment[kRight].pts[kTail]);
  } else if (BoundSkewTree::isParallel(left, right)) {
    auto min_x = fmax(fmin(left.pts[kHead].x, left.pts[kTail].x), fmin(right.pts[kHead].x, right.pts[kTail].x));
    auto max_x = fmin(fmax(left.pts[kHead].x, left.pts[kTail].x), fmax(right.pts[kHead].x, right.pts[kTail].x));
    auto min_y = fmax(fmin(left.pts[kHead].y, left.pts[kTail].y), fmin(right.pts[kHead].y, right.pts[kTail].y));
    auto max_y = fmin(fmax(left.pts[kHead].y, left.pts[kTail].y), fmax(right.pts[kHead].y, right.pts[kTail].y));
    if ((left_type == LineType::kVertical || left_type == LineType::kTilt) && max_y >= min_y) {
      BoundSkewTree::calcCoord(_join_segment[kLeft].pts[kHead], left, min_y);
      BoundSkewTree::calcCoord(_join_segment[kLeft].pts[kTail], left, max_y);
      BoundSkewTree::calcCoord(_join_segment[kRight].pts[kHead], right, min_y);
      BoundSkewTree::calcCoord(_join_segment[kRight].pts[kTail], right, max_y);
    } else if ((left_type == LineType::kHorizontal || left_type == LineType::kFlat) && max_x >= min_x) {
      BoundSkewTree::calcCoord(_join_segment[kLeft].pts[kHead], left, min_x);
      BoundSkewTree::calcCoord(_join_segment[kLeft].pts[kTail], left, max_x);
      BoundSkewTree::calcCoord(_join_segment[kRight].pts[kHead], right, min_x);
      BoundSkewTree::calcCoord(_join_segment[kRight].pts[kTail], right, max_x);
    }
  } else {
    // single point case
  }
  if (BoundSkewTree::lineType(getJsLine(_join_segment,kLeft)) == LineType::kManhattan && left_type != LineType::kManhattan && right_type != LineType::kManhattan) {
    _ms[kLeft].makeDiamond(closest[kLeft], 0);
    _ms[kRight].makeDiamond(closest[kRight], 0);
  }
  // checkUpdateJs(cur, left, right);
}
__device__ double calcJrArea(const Line& l1, const Line& l2) {
  auto min_x = fmin(fmin(l1.pts[kHead].x, l1.pts[kTail].x), fmin(l2.pts[kHead].x, l2.pts[kTail].x));
  auto max_x = fmax(fmax(l1.pts[kHead].x, l1.pts[kTail].x), fmax(l2.pts[kHead].x, l2.pts[kTail].x));
  auto min_y = fmin(fmin(l1.pts[kHead].y, l1.pts[kTail].y), fmin(l2.pts[kHead].y, l2.pts[kTail].y));
  auto max_y = fmax(fmax(l1.pts[kHead].y, l1.pts[kTail].y), fmax(l2.pts[kHead].y, l2.pts[kTail].y));
  auto bound_area = (max_x - min_x) * (max_y - min_y);
  auto tri_area_1 = 0.5 * fabs(l1.pts[kHead].x - l1.pts[kTail].x) * fabs(l1.pts[kHead].y - l1.pts[kTail].y);
  auto tri_area_2 = 0.5 * fabs(l2.pts[kHead].x - l2.pts[kTail].x) * fabs(l2.pts[kHead].y - l2.pts[kTail].y);
  auto jr_area = bound_area - tri_area_1 - tri_area_2;
  return jr_area;
}
__device__ double ptSkew(const Pt& pt){
  return pt.max - pt.min;
}
__device__ void calcBsLocated(Area* cur, Pt& pt, Line& result_line) {
    // 1. 直接遍历mr.pts，避免填充临时数组
    for (int i = 0; i < cur->mr.size; ++i) {
        int j = (i + 1) % cur->mr.size;
        Line line(cur->mr.pts[i], cur->mr.pts[j]);
        
        // 2. 提前终止条件
        if (BoundSkewTree::onLine(pt, line)) {
            result_line = line;
            return; // 找到匹配线段
        }
    }
}
__device__ void calcIrregularPtDelays(Area* cur, Pt& pt, Line& line){
  auto x = fabs(line.pts[kHead].x - line.pts[kTail].x);
  auto y = fabs(line.pts[kHead].y - line.pts[kTail].y);
  auto left_line = cur->lines[kLeft];
  auto right_line = cur->lines[kRight];
  auto js_type = BoundSkewTree::lineType(cur->lines[kLeft]);
  if (js_type == LineType::kManhattan) {
    // LOG_FATAL_IF(!isSame(left_line[kHead], left_line[kTail]) || !isSame(right_line[kHead], right_line[kTail]))
    //     << "endpoint should be same, left head: [" << left_line[kHead].x << ", " << left_line[kHead].y << "], left tail: ["
    //     << left_line[kTail].x << ", " << left_line[kTail].y << "], right head: [" << right_line[kHead].x << ", " << right_line[kHead].y
    //     << "], right tail: [" << right_line[kTail].x << ", " << right_line[kTail].y << "]";

    auto delay_left = ptDelayIncrease(left_line.pts[kHead], pt, cur->get_left->cap, _pattern);
    auto delay_right = ptDelayIncrease(right_line.pts[kHead], pt, cur->get_right->cap, _pattern);
    pt.min = fmin(left_line.pts[kHead].min + delay_left, right_line.pts[kHead].min + delay_right);
    pt.max = fmax(left_line.pts[kHead].max + delay_left, right_line.pts[kHead].max + delay_right);
    // LOG_FATAL_IF(ptSkew(pt) >= _skew_bound + kEpsilon) << "skew is larger than skew bound";
  } else {
    // LOG_FATAL_IF(js_type != LineType::kVertical && js_type != LineType::kHorizontal) << "js type is not vertical or horizontal";
    auto dist = BoundSkewTree::distanceR(pt, line.pts[kHead]);
    auto length = x + y;
    double alpha = 0;
    if (x > y) {
      auto m = y / x;
      auto ratio = pow(1 + fabs(m), 2);
      alpha = (_K[kH] + m * m * _K[kV]) / ratio;
    } else {
      auto m = x / y;
      auto ratio = pow(1 + fabs(m), 2);
      alpha = (_K[kV] + m * m * _K[kH]) / ratio;
    }
    auto beta = (line.pts[kTail].max - line.pts[kHead].max) / length - alpha * length;
    pt.max = line.pts[kHead].max + alpha * dist * dist + beta * dist;
    beta = (line.pts[kTail].min - line.pts[kHead].min) / length - alpha * length;
    pt.min = line.pts[kHead].min + alpha * dist * dist + beta * dist;
  }
}
__device__ void checkPtDelay(Pt& pt){
  // LOG_ERROR_IF(pt.min <= -kEpsilon) << "pt min delay is negative";
  // LOG_FATAL_IF(pt.max - pt.min <= -kEpsilon) << "pt skew is negative";
  if (pt.min < -kEpsilon) {
    pt.min = 0;
  }
  if (pt.max < pt.min + kEpsilon) {
    pt.max = pt.min;
  }
}
__device__ void calcPtDelays(Area* cur, Pt& pt, Line& line){
  // LOG_FATAL_IF(!Geom::onLine(pt, line)) << "point is not located in line";
  auto dist = BoundSkewTree::distanceR(pt, line.pts[kHead]);
  auto x = fabs(line.pts[kHead].x - line.pts[kTail].x);
  auto y = fabs(line.pts[kHead].y - line.pts[kTail].y);
  auto length = x + y;
  // printf("calcPtDelays:%f,%f,%f,%f,%f\n",dist,x,y,line.pts[kHead].min,line.pts[kHead].max);
  if (BoundSkewTree::Equal(dist, 0)) {
    pt.min = line.pts[kHead].min;
    pt.max = line.pts[kHead].max;
    // printf("min:max,%f%f\n",pt.min,pt.max);
  } else if (BoundSkewTree::isSame(pt, line.pts[kTail])) {
    pt.min = line.pts[kTail].min;
    pt.max = line.pts[kTail].max;
  } else if (BoundSkewTree::Equal(x, y)) {
    // line is manhattan arc
    // LOG_FATAL_IF(!Equal(line[kHead].min, line[kTail].min) || !Equal(line[kHead].max, line[kTail].max))
    //     << "manhattan arc endpoint's delay is not same";
    pt.min = line.pts[kHead].min = line.pts[kTail].min;
    pt.max = line.pts[kHead].max = line.pts[kTail].max;
  } else if (BoundSkewTree::Equal(x, 0) || BoundSkewTree::Equal(y, 0)) {
    // line is vertical or horizontal
    auto alpha = BoundSkewTree::Equal(x, 0) ? _K[kV] : _K[kH];
    auto beta = (line.pts[kTail].min - line.pts[kHead].min) / length - alpha * length;
    pt.min = line.pts[kHead].min + alpha * dist * dist + beta * dist;
    beta = (line.pts[kTail].max - line.pts[kHead].max) / length - alpha * length;
    pt.max = line.pts[kHead].max + alpha * dist * dist + beta * dist;
  } else {
    // LOG_FATAL_IF(!cur) << "cur is nullptr";
    // LOG_FATAL_IF(!Equal(ptSkew(line[kHead]), _skew_bound) || !Equal(ptSkew(line[kTail]), _skew_bound))
    //     << "thera are skew reservation in line";
    calcIrregularPtDelays(cur, pt, line);
  }
  // printf("min:max,%f%f\n",pt.min,pt.max);
  checkPtDelay(pt);
}
__device__ void calcJsDelay(Area* left, Area* right,const Side<Pts>& _join_segment){
  FOR_EACH_SIDE(left_side)
  {
    Line line;
    calcBsLocated(left, _join_segment[kLeft].pts[left_side], line);
    // printf("calcJsDelay:%f,%f\n",line.pts[0].min,line.pts[1].min);
    calcPtDelays(left, _join_segment[kLeft].pts[left_side], line);
  }
  FOR_EACH_SIDE(right_side)
  {
    Line line;
    calcBsLocated(right, _join_segment[kRight].pts[right_side], line);
    // printf("calcJsDelay:%f,%f\n",line.pts[0].min,line.pts[1].min);
    calcPtDelays(right, _join_segment[kRight].pts[right_side], line);
  }
}
__device__ void initSide(Side<Pts>& join_segment, Side<Pts>& join_region){
  FOR_EACH_SIDE(side)
  {
    join_region[side].size=0;
    join_segment[side].size=0;
  }
}
__device__ void getConvexHullLines(Line* lines,Region convex_hull){
  for (size_t i = 0; i < convex_hull.size; ++i) {
      auto j = (i + 1) % convex_hull.size;
      lines[i]=Line(convex_hull.pts[i],convex_hull.pts[j]);
    }
}
__device__ void checkJsMs(const Side<Pts>& join_segment) {
  Trr left, right;
  auto left_js = getJsLine(join_segment,kLeft);
  auto right_js = getJsLine(join_segment,kRight);
  BoundSkewTree::lineToMs(left, left_js);
  BoundSkewTree::lineToMs(right, right_js);
  // LOG_FATAL_IF(!isTrrContain(left, _ms[kLeft])) << "left js is not contain in left ms";
  // LOG_FATAL_IF(!isTrrContain(right, _ms[kRight])) << "right js is not contain in right ms";
}
__device__ void calcJS(Area* cur, Line& left, Line& right,Side<Pts>& join_segment,Side<Trr>& ms){
  PtPair closest;
  auto line_dist = BoundSkewTree::lineDist(left, right, closest);
  // printf("lineDist%d\n",join_segment[0].size);
  auto left_js_bak = getJsLine(join_segment,kLeft);
  auto right_js_bak = getJsLine(join_segment,kRight);
  auto left_ms_bak = ms[kLeft];
  auto right_ms_bak = ms[kRight];
  // printf("ms[kLeft]._x_low: %f\n",ms[kLeft]._x_low);
  if (BoundSkewTree::Equal(line_dist, cur->radius)) {
    cur->radius=line_dist;
    updateJS(cur, left, right, closest,join_segment,ms);
    auto origin_area = calcJrArea(left_js_bak, right_js_bak);
    auto new_area = calcJrArea(getJsLine(join_segment,kLeft), getJsLine(join_segment,kRight));
    if (origin_area >= new_area) {
      setJsLine(kLeft, left_js_bak,join_segment);
      setJsLine(kRight, right_js_bak,join_segment);
      if (BoundSkewTree::lineType(left_js_bak) == LineType::kManhattan) {
        ms[kLeft] = left_ms_bak;
        ms[kRight] = right_ms_bak;
      }
    }
  } else if (line_dist < cur->radius) {
    cur->radius=line_dist;
    updateJS(cur, left, right, closest,join_segment,ms);
  }
  if (BoundSkewTree::lineType(getJsLine(join_segment,kLeft)) == LineType::kManhattan) {
    checkJsMs(join_segment);
  }
}
__global__ void calcJS(Area*areas,int beginID,int areasize,Side<Pts>* join_segment, Side<Pts>* join_region,Side<Trr>* ms){
    int id = blockIdx.x * blockDim.x + threadIdx.x; 
  if(id>=areasize)return;
  int size=areas[id+beginID].r-areas[id+beginID].l+1;
  if(size<=1)return;
  Area* parent=&areas[id+beginID];
  Area* left=&areas[areas[id+beginID].left];
  Area* right=&areas[areas[id+beginID].right];
  initSide(join_segment[id],join_region[id]);
  parent->radius=DOUBLE_MAX;
  Line left_lines[32];
  getConvexHullLines(left_lines,left->convex_hull);
  Line right_lines[32];
  getConvexHullLines(right_lines,right->convex_hull);
  // printf("left_and_convex_hull_size,%d,%d\n",left->convex_hull.size,right->convex_hull.size);
  // printf("\njoin_segment_size:%d,%d\n",join_segment[id][0].size,join_segment[id][1].size);
  for(int i=0;i<left->convex_hull.size;i++){
    for(int j=0;j<right->convex_hull.size;j++){
      PtPair closest;
      BoundSkewTree::lineDist(left_lines[i], right_lines[j], closest);
      calcJS(parent, left_lines[i], right_lines[j],join_segment[id],ms[id]);
    }
  }
  calcJsDelay(left, right,join_segment[id]);
  if (BoundSkewTree::lineType(getJsLine(join_segment[id],kLeft)) == LineType::kManhattan) {//bug
    checkJsMs(join_segment[id]);
  }
}
__device__ RelativeType lineRelative(const Line& l1, const Line& l2, const size_t& ref){
  // return the position of one line relative to ref line
  auto line_type = BoundSkewTree::lineType(l1);
  auto t_line_type = BoundSkewTree::lineType(l2);
  // LOG_FATAL_IF(line_type != t_line_type) << "line type is not same";
  if (line_type == LineType::kVertical || line_type == LineType::kTilt) {
    auto max_x1 = fmax(l1.pts[kHead].x, l1.pts[kTail].x);
    auto max_x2 = fmax(l2.pts[kHead].x, l2.pts[kTail].x);
    if ((max_x1 <= max_x2 && ref == kLeft) || (max_x1 >= max_x2 && ref == kRight)) {
      return RelativeType::kLeft;
    } else {
      return RelativeType::kRight;
    }
  } else if (line_type == LineType::kHorizontal || line_type == LineType::kFlat) {
    auto max_y1 = fmax(l1.pts[kHead].y, l1.pts[kTail].y);
    auto max_y2 =fmax(l2.pts[kHead].y, l2.pts[kTail].y);
    if ((max_y1 <= max_y2 && ref == kLeft) || (max_y1 >= max_y2 && ref == kRight)) {
      return RelativeType::kBottom;
    } else {
      return RelativeType::kTop;
    }
  } else if (line_type == LineType::kManhattan) {
    return RelativeType::kManhattanParallel;
  } else {
    // LOG_FATAL << "line type error" << std::endl;
  }
}
__device__ void calcRelativeCoord(Pt& p, const RelativeType& type, const double& shift){
  switch (type) {
    case RelativeType::kLeft:
      p.x += shift;
      break;
    case RelativeType::kRight:
      p.x -= shift;
      break;
    case RelativeType::kTop:
      p.y -= shift;
      break;
    case RelativeType::kBottom:
      p.y += shift;
      break;
    default:
      break;
  }
}
__device__ void addJsPts(Area* parent, Area* left, Area* right,Side<Pts>& _join_segment){
  // add points on origin js lines
  FOR_EACH_SIDE(side)
  {
    // LOG_FATAL_IF(isSame(_join_segment[side][kHead], _join_segment[side][kTail])) << "join segment is a point";
    auto mr = side == kLeft ? left->mr : right->mr;
    // printf("mr:%d",mr.size);
    for(int i=0;i<mr.size;i++){
      if (BoundSkewTree::onLine(mr.pts[i], getJsLine(_join_segment,side)) && !BoundSkewTree::isSame(mr.pts[i], _join_segment[side].pts[kHead])
          && !BoundSkewTree::isSame(mr.pts[i], _join_segment[side].pts[kTail])) {
        // _join_segment[side].push_back(mr.pts[i]);
         _join_segment[side].pts[_join_segment[side].size++]=mr.pts[i];  //++
      }
    }
    sortPtsByFront(_join_segment[side]);
  }
  // add points on other side
  // auto new_js = _join_segment;
  Pt new_js[2][16];
  int cnt[2]={0,0};
  FOR_EACH_SIDE(side){
    for(int i=0;i<_join_segment[side].size;i++)new_js[side][cnt[side]++]=_join_segment[side].pts[i];
  }
  FOR_EACH_SIDE(side)
  {
    auto other_side = side == kLeft ? kRight : kLeft;
    auto other_mr = other_side == kLeft ? left->mr : right->mr;
    auto relative_type = lineRelative(getJsLine(_join_segment,kLeft), getJsLine(_join_segment,kRight), other_side);
    // for (auto pt : other_mr) {
    for(int k=0;k<other_mr.size;k++){
      calcRelativeCoord(other_mr.pts[k], relative_type, parent->radius);
      for (size_t i = 0; i < _join_segment[side].size - 1; ++i) {
        Line line = {_join_segment[side].pts[i], _join_segment[side].pts[i + 1]};
        if (BoundSkewTree::onLine(other_mr.pts[k], line) && !BoundSkewTree::isSame(other_mr.pts[k], _join_segment[side].pts[i]) && !BoundSkewTree::isSame(other_mr.pts[k], _join_segment[side].pts[i + 1])) {
          calcPtDelays(nullptr, other_mr.pts[k], line);
          // new_js[side].push_back(pt);
          new_js[side][cnt[side]++]=other_mr.pts[k];
          break;
        }
      }
    }
    sortPtsByFront(new_js[side],cnt[side]);
  }
  FOR_EACH_SIDE(side)
  {
    for(int i=0;i<cnt[side];i++)
    _join_segment[side].pts[i] = new_js[side][i];
    _join_segment[side].size=cnt[side];
  }
}
__device__ double delayFromJs(const size_t& js_side, const size_t& side, const size_t& idx, const size_t& timing_type,
                                  const Side<double>& delay_from,Side<Pts>&_join_segment) {
  double delay = timing_type == kMin ? _join_segment[side].pts[idx].min : _join_segment[side].pts[idx].max;
  delay += js_side == side ? 0 : delay_from[side];
  return delay;
}
__device__ void addTurnPt(const size_t& side, const size_t& idx, const size_t& timing_type, const Side<double>& delay_from,Side<Pts>&_join_segment,Side<Pts>& _join_region){
  auto p1 = _join_region[side].pts[idx];
  auto p2 = _join_region[side].pts[idx + 1];
  double alpha = 0;
  if (BoundSkewTree::Equal(p1.x, p2.x)) {
    alpha = _K[kV];
  } else {
    alpha = _K[kH];
  }
  auto dist = BoundSkewTree::distanceR(p1, p2);
  // LOG_FATAL_IF(Equal(dist, 0)) << "distance is zero";
  Side<Side<double>> beta = { {0, 0}, {0, 0} };
  FOR_EACH_SIDE(sub_side)  // left and right
  {
    FOR_EACH_SIDE(timing_side)  // min and max
    {
      auto t1 = delayFromJs(side, sub_side, idx, timing_side, delay_from,_join_segment);
      auto t2 = delayFromJs(side, sub_side, idx + 1, timing_side, delay_from,_join_segment);
      beta[sub_side][timing_side] = (t2 - t1) / dist - alpha * dist;
    }
  }
  auto turn_dist = (delayFromJs(side, kLeft, idx, timing_type, delay_from,_join_segment) - delayFromJs(side, kRight, idx, timing_type, delay_from,_join_segment))
                   / (beta[kRight][timing_type] - beta[kLeft][timing_type]);
  // LOG_FATAL_IF(turn_dist <= 0 || turn_dist >= dist) << "turn dist is not in range";
  auto ref_dist = dist - turn_dist;
  Pt turn_pt((p1.x * ref_dist + p2.x * turn_dist) / dist, (p1.y * ref_dist + p2.y * turn_dist) / dist);
  Side<Side<double>> delay_bound = { {0, 0}, {0, 0} };
  FOR_EACH_SIDE(sub_side)  // left and right
  {
    FOR_EACH_SIDE(timing_side)  // min and max
    {
      delay_bound[sub_side][timing_side] = delayFromJs(side, sub_side, idx, timing_side, delay_from,_join_segment) + alpha * turn_dist * turn_dist
                                           + beta[sub_side][timing_side] * turn_dist;
    }
  }
  turn_pt.min = fmin(delay_bound[kLeft][kMin], delay_bound[kRight][kMin]);
  turn_pt.max = fmax(delay_bound[kLeft][kMax], delay_bound[kRight][kMax]);
  // _join_region[side].push_back(turn_pt);
  _join_region[side].pts[_join_region[side].size++]=turn_pt;
}
__device__ void calcNotManhattanJrEndpoints(Area* parent, Area* left, Area* right,Side<Pts>& _join_segment, Side<Pts>& _join_region){
  addJsPts(parent, left, right,_join_segment);
  // printf("calcNotManhattanJrEndpoints %d\n",_join_segment[0].size);
  Side<double> delay_from = {ptDelayIncrease(_join_segment[kLeft].pts[kHead], _join_segment[kRight].pts[kHead], left->cap, _pattern),
                             ptDelayIncrease(_join_segment[kLeft].pts[kHead], _join_segment[kRight].pts[kHead], right->cap, _pattern)};
  FOR_EACH_SIDE(side)
  {
    auto other_side = side == kLeft ? kRight : kLeft;
    // _join_region[side] = _join_segment[side];
    _join_region[side].size=_join_segment[side].size;
    // for(int i=0;i<_join_segment[side].size;i++)_join_region[side].pts[i]=_join_segment[side].pts[i];
    for (size_t i = 0; i < _join_segment[side].size; ++i) {
      auto pt = _join_segment[side].pts[i];
      pt.min = fmin(pt.min, _join_segment[other_side].pts[i].min + delay_from[other_side]);
      pt.max = fmax(pt.max, _join_segment[other_side].pts[i].max + delay_from[other_side]);
      _join_region[side].pts[i] = pt;
    }
    uniquePtsLoc(_join_region[side]);
  }
  // printf("calcNotManhattanJrEndpoints1 %d\n",_join_segment[1].size);
  // for(int i=0;i<_join_segment[0].size;i++)printf("_join_segment[0]:%f %f\n",_join_segment[0].pts[i].x,_join_segment[0].pts[i].y);
  // for(int i=0;i<_join_segment[1].size;i++)printf("_join_segment[1]:%f %f\n",_join_segment[1].pts[i].x,_join_segment[1].pts[i].y);
  // for(int i=0;i<_join_region[0].size;i++)printf("_join_region[0]:%f %f\n",_join_region[0].pts[i].x,_join_region[0].pts[i].y);
  // for(int i=0;i<_join_region[1].size;i++)printf("_join_region[1]:%f %f\n",_join_region[1].pts[i].x,_join_region[1].pts[i].y);
  FOR_EACH_SIDE(side)
  {
    auto other_side = side == kLeft ? kRight : kLeft;
    // add JR turn points which delay slope is changed
    auto n = _join_region[side].size - 1;
    for (size_t i = 0; i < n; ++i) {
      auto delta = (_join_segment[side].pts[i].min - _join_segment[other_side].pts[i].min - delay_from[other_side])
                   * (_join_segment[side].pts[i + 1].min - _join_segment[other_side].pts[i + 1].min - delay_from[other_side]);
      if (delta < -kEpsilon) {
        addTurnPt(side, i, kMin, delay_from,_join_segment,_join_region);
      }
      delta = (_join_segment[side].pts[i].max - _join_segment[other_side].pts[i].max - delay_from[other_side])
              * (_join_segment[side].pts[i + 1].max - _join_segment[other_side].pts[i + 1].max - delay_from[other_side]);
      if (delta < -kEpsilon) {
        addTurnPt(side, i, kMax, delay_from,_join_segment,_join_region);
      }
    }
    sortPtsByFront(_join_region[side]);
    uniquePtsLoc(_join_region[side]);
  }
  // printf("calcNotManhattanJrEndpoints2\n");
  // if(parent->x>46&&parent->x<47)for(int i=0;i<_join_region[1].size;i++)printf("_join_region[1]:%f %f %f %f\n",_join_region[1].pts[i].x,_join_region[1].pts[i].y,_join_region[1].pts[i].min,_join_region[1].pts[i].max);
  FOR_EACH_SIDE(side)
  {
    // remove redundant turn points which have same slope
    for (size_t i = 0; i < _join_region[side].size - 1; ++i) {
      auto pt1 = _join_region[side].pts[i];
      auto pt2 = _join_region[side].pts[i + 1];
      auto dist = BoundSkewTree::distanceR(pt1, pt2);
      // LOG_FATAL_IF(Equal(dist, 0)) << "distance is zero";
      _join_region[side].pts[i].val = (ptSkew(pt2) - ptSkew(pt1)) / dist;
    }
    
    // remove redundant turn points which skew slope is not strictly monotone increasing
    // Pts incr_pts = {_join_region[side].pts[0]};
    Pt incr_pts[16];
    int cnt=0;
    incr_pts[cnt++]=_join_region[side].pts[0];
    for (size_t j = 1; j < _join_region[side].size - 1; ++j) {
      auto cur_val = incr_pts[cnt-1].val;
      auto next_val = _join_region[side].pts[j].val;
      // LOG_FATAL_IF(cur_val > next_val + 100 * kEpsilon)
      //     << "cur_val: " << cur_val << "> next_val: " << next_val << ", skew slope is not strictly monotone increasing";
      if (next_val > cur_val + kEpsilon) {//精度问题
        // if(side==1&&parent->x>46&&parent->x<47)printf("%f %f %f %f\n",next_val,cur_val,_join_region[side].pts[j].x,_join_region[side].pts[j].y);
        // incr_pts.push_back(_join_region[side].pts[j]);
        incr_pts[cnt++]=_join_region[side].pts[j];
      }
    }
    // incr_pts.push_back(_join_region[side].back());
    incr_pts[cnt++]=_join_region[side].pts[_join_region[side].size-1];
    // _join_region[side] = incr_pts;
    for(int i=0;i<cnt;i++){_join_region[side].pts[i]=incr_pts[i];}
    _join_region[side].size=cnt;
  }
    // if(parent->x>46&&parent->x<47)for(int i=0;i<_join_region[1].size;i++)printf("_join_region[1]:%f %f %f %f\n",_join_region[1].pts[i].x,_join_region[1].pts[i].y,_join_region[1].pts[i].min,_join_region[1].pts[i].max);
}
__device__ void jsProcess(Area* cur,Side<Pts>& _join_segment,Side<Pts>& _join_region){
  auto swap = [](Pt& p1, Pt& p2) {
    auto temp = p1;
    p1 = p2;
    p2 = temp;
  };
  FOR_EACH_SIDE(side)
  {
    if (BoundSkewTree::Equal(_join_segment[side].pts[kHead].y, _join_segment[side].pts[kTail].y)) {
      if (_join_segment[side].pts[kHead].x < _join_segment[side].pts[kTail].x) {
        swap(_join_segment[side].pts[kHead], _join_segment[side].pts[kTail]);
      }
    } else if (_join_segment[side].pts[kHead].y < _join_segment[side].pts[kTail].y) {
      swap(_join_segment[side].pts[kHead], _join_segment[side].pts[kTail]);
    }
  }
  FOR_EACH_SIDE(side)
  {
    setJrLine(side, getJsLine(_join_segment,side),_join_region);
    cur->lines[side]=getJsLine(_join_segment,side);
  }
}
// __global__ void jsProcess(Area*areas,int beginID,int areasize,Side<Pts>* _join_segment,Side<Pts>* _join_region){
//   int id = blockIdx.x * blockDim.x + threadIdx.x; 
//   if(id>=areasize)return;
//   int size=areas[id+beginID].r-areas[id+beginID].l+1;
//   if(size<=1)return;
//   Area* cur=&areas[id+beginID];
//   auto swap = [](Pt& p1, Pt& p2) {
//     auto temp = p1;
//     p1 = p2;
//     p2 = temp;
//   };
//   FOR_EACH_SIDE(side)
//   {
//     if (Equal(_join_segment[id][side].pts[kHead].y, _join_segment[id][side].pts[kTail].y)) {
//       if (_join_segment[id][side].pts[kHead].x < _join_segment[id][side].pts[kTail].x) {
//         swap(_join_segment[id][side].pts[kHead], _join_segment[id][side].pts[kTail]);
//       }
//     } else if (_join_segment[id][side].pts[kHead].y < _join_segment[id][side].pts[kTail].y) {
//       swap(_join_segment[id][side].pts[kHead], _join_segment[id][side].pts[kTail]);
//     }
//   }
//   FOR_EACH_SIDE(side)
//   {
//     setJrLine(side, getJsLine(_join_segment[id],side),_join_region[id]);
//     cur->lines[side]=getJsLine(_join_segment[id],side);
//   }
// }
__device__ LineType calcAreaLineType(Area* cur){
  // printf("calcAreaLineType:%f %f %f %f\n",cur->lines[kLeft].pts[0].x,cur->lines[kLeft].pts[0].y,cur->lines[kLeft].pts[1].x,cur->lines[kLeft].pts[1].y);//check
  auto line = cur->lines[kLeft];
  return BoundSkewTree::lineType(line);
}
__device__ void updatePtDelaysByEndSide(Area* cur, const size_t& end_side, Pt& pt){
  auto left_line = cur->lines[kLeft];
  auto right_line = cur->lines[kRight];
  // printf("updatePtDelaysByEndSide:%f\n",left_line.pts[0].x);
  // printf("cur->get_left->cap:%f\n",cur->get_left->cap);
  auto delay_left = ptDelayIncrease(pt, left_line.pts[end_side], cur->get_left->cap, _pattern);
  auto delay_right = ptDelayIncrease(pt, right_line.pts[end_side], cur->get_right->cap, _pattern);
  pt.min = fmin(left_line.pts[end_side].min + delay_left, right_line.pts[end_side].min + delay_right);
  pt.max = fmax(left_line.pts[end_side].max + delay_left, right_line.pts[end_side].max + delay_right);
}
__device__ void addFmsToJr(Side<Pts>& _join_region){
  FOR_EACH_SIDE(side)
  {
    for (size_t i = 0; i < _join_region[side].size - 1; ++i) {
      auto pt_cur = _join_region[side].pts[i];
      auto pt_next = _join_region[side].pts[i + 1];
      auto delta_cur = ptSkew(pt_cur) - _skew_bound;
      auto delta_next = ptSkew(pt_next) - _skew_bound;
      if (delta_cur * delta_next < 0 && !BoundSkewTree::Equal(delta_cur, 0) && !BoundSkewTree::Equal(delta_next, 0)) {
        auto dist = BoundSkewTree::distanceR(pt_cur, pt_next);
        auto turn_dist = (_skew_bound - ptSkew(pt_cur)) * dist / (ptSkew(pt_next) - ptSkew(pt_cur));
        auto ref_dist = dist - turn_dist;
        Pt turn_pt{(pt_cur.x * ref_dist + pt_next.x * turn_dist) / dist, (pt_cur.y * ref_dist + pt_next.y * turn_dist) / dist};
        Line line = {pt_cur, pt_next};
        calcPtDelays(nullptr, turn_pt, line);
        // _join_region[side].insert(_join_region[side].begin() + i + 1, turn_pt);
        for (int j = _join_region[side].size; j > i + 1; --j) {_join_region[side].pts[j] = _join_region[side].pts[j-1];}
        _join_region[side].pts[i+1]=turn_pt;
        _join_region[side].size++;
      }
    }
  }
}
__device__ void calcJrEndpoints(Area* cur,Side<Pts>& _join_segment,Side<Pts>& _join_region){
  auto left_line = cur->lines[kLeft];
  auto right_line = cur->lines[kRight];
  // printf("calcJrEndpoints:%f\n",left_line.pts[0].x);
  // LOG_FATAL_IF(!Geom::isSame(_join_segment[kLeft][kHead], left_line[kHead]) || !Geom::isSame(_join_segment[kLeft][kHead], left_line[kHead]))
  //     << "left join segment is not same as left line at head";
  // LOG_FATAL_IF(!Geom::isSame(_join_segment[kLeft][kTail], left_line[kTail]) || !Geom::isSame(_join_segment[kLeft][kTail], left_line[kTail]))
  //     << "left join segment is not same as left line at tail";
  // LOG_FATAL_IF(!Geom::isSame(_join_segment[kRight][kHead], right_line[kHead])
  //              || !Geom::isSame(_join_segment[kRight][kHead], right_line[kHead]))
  //     << "right join segment is not same as right line at head";
  // LOG_FATAL_IF(!Geom::isSame(_join_segment[kRight][kTail], right_line[kTail])
  //              || !Geom::isSame(_join_segment[kRight][kTail], right_line[kTail]))
  //     << "right join segment is not same as right line at tail";
  // printf("_join_region[kLeft]_size%d\n",_join_region[kLeft].size);
  _join_region[kLeft].pts[kHead] = _join_segment[kLeft].pts[kHead];
  _join_region[kLeft].pts[kTail] = _join_segment[kLeft].pts[kTail];
  _join_region[kRight].pts[kHead] = _join_segment[kRight].pts[kHead];
  _join_region[kRight].pts[kTail] = _join_segment[kRight].pts[kTail];
  // printf("_join_region[kLeft]_size:%f\n",_join_region[kLeft].pts[kHead].x);
  updatePtDelaysByEndSide(cur, kHead, _join_region[kLeft].pts[kHead]);
  updatePtDelaysByEndSide(cur, kHead, _join_region[kRight].pts[kHead]);
  updatePtDelaysByEndSide(cur, kTail, _join_region[kLeft].pts[kTail]);
  updatePtDelaysByEndSide(cur, kTail, _join_region[kRight].pts[kTail]);
  // printf("updatePtDelaysByEndSide\n");
}
__device__ void calcJr(int id,int beginID,Area* parent, Area* left, Area* right,Side<Pts>& _join_segment, Side<Pts>& _join_region){
  // printf("calcJr begin\n");
  // printf("%f\n",parent->lines[0].pts[0].x);
  if (calcAreaLineType(parent) == LineType::kManhattan) {
    // printf("if (calcAreaLineType(parent) == LineType::kManhattan)\n");
    calcJrEndpoints(parent,_join_segment,_join_region);
    // printf("Yes\n");
  } else {
    // printf("No\n");
    calcNotManhattanJrEndpoints(parent, left, right,_join_segment,_join_region);
  }
  addFmsToJr(_join_region);
  // printf("%f\n",parent->lines[0].pts[0].x);
}
__device__ bool jrCornerExist(const size_t& end_side,Side<Pts>&_join_segment){
  auto pt1 = _join_segment[kLeft].pts[end_side];
  auto pt2 = _join_segment[kRight].pts[end_side];
  return !BoundSkewTree::Equal(pt1.x, pt2.x) && !BoundSkewTree::Equal(pt1.y, pt2.y);
}
__device__ void calcJrCorner(Area* cur,Side<Pts>&_join_segment,Side<Pt>&_join_corner){
  FOR_EACH_SIDE(side)
  {
    // LOG_FATAL_IF(_join_segment[side].front().y + kEpsilon < _join_segment[side].back().y) << "join segment direction is not correct";
  }
  // printf("%f\n",cur->lines[0].pts[0].x);
  // printf("calcJrCorner:%d\n",calcAreaLineType(cur));
  if (calcAreaLineType(cur) == LineType::kManhattan && !BoundSkewTree::Equal(cur->radius, 0)) {
    FOR_EACH_SIDE(end_side)
    {
      if (jrCornerExist(end_side,_join_segment)) {
        auto p_left = _join_segment[kLeft].pts[end_side];
        auto p_right = _join_segment[kRight].pts[end_side];
        if ((p_left.x - p_right.x) * (p_left.y - p_right.y) < 0) {
          if (end_side == kHead) {
            _join_corner[end_side] = {fmax(p_left.x, p_right.x), fmax(p_left.y, p_right.y)};
          } else {
            _join_corner[end_side] = {fmin(p_left.x, p_right.x), fmin(p_left.y, p_right.y)};
          }
        } else {
          if (end_side == kHead) {
            _join_corner[end_side] = {fmin(p_left.x, p_right.x), fmax(p_left.y, p_right.y)};
          } else {
            _join_corner[end_side] = {fmax(p_left.x, p_right.x), fmin(p_left.y, p_right.y)};
          }
        }
        updatePtDelaysByEndSide(cur, end_side, _join_corner[end_side]);
      }
    }
  }
}
__device__ void calcMergeDist(const double& r, const double& c, const double& cap1, const double& delay1, const double& cap2,
                                  const double& delay2, const double& dist, double& d1, double& d2){
  auto target = (delay2 - delay1 + r * dist * (cap2 + c * dist / 2)) / (r * (cap1 + cap2 + c * dist));
  // printf("calcMergeDist:%f,%f\n",target,dist);
  if (target < 0) {
    auto fact = cap2 / c;
    target = sqrt(fact * fact + 2 * (delay1 - delay2) / (r * c)) - fact;
    d1 = 0;
    d2 = target;
  } else if (target > dist) {
    auto fact = cap1 / c;
    target = sqrt(fact * fact + 2 * (delay2 - delay1) / (r * c)) - fact;
    d1 = target;
    d2 = 0;
  } else {
    d1 = target;
    d2 = dist - target;
  }
  // printf("r=%.6e, c=%.6e, cap1=%.6e, delay1=%.6e, cap2=%.6e, delay2=%.6e, dist=%.6f,%f\n",
  //      r, c, cap1, delay1, cap2, delay2, dist,target);
}
__device__ void calcPtCoordOnLine(const Pt& p1, const Pt& p2, const double& d1, const double& d2, Pt& pt){
  auto dist = d1 + d2;
  auto pt_dist = BoundSkewTree::distanceR(p1, p2);
  // LOG_FATAL_IF(!Equal(dist, pt_dist) && dist < pt_dist) << "dist is less than points dist";
  if (BoundSkewTree::Equal(d1, 0)) {
    pt = p1;
  } else if (BoundSkewTree::Equal(d2, 0)) {
    pt = p2;
  } else {
    pt = {(p1.x * d2 + p2.x * d1) / dist, (p1.y * d2 + p2.y * d1) / dist};
  }
}
__device__ void calcBalPtOnLine(Pt& p1, Pt& p2, const size_t& timing_type, double& d1, double& d2, Pt& bal_pt,
                                    const RCPattern& pattern) {
  auto h = fabs(p1.x - p2.x);
  auto v = fabs(p1.y - p2.y);
  // LOG_FATAL_IF(!Equal(h, 0) && !Equal(v, 0)) << "h and v are not zero, which balance point is not on line";

  auto delay1 = timing_type == kMin ? p1.min : p1.max;
  auto delay2 = timing_type == kMin ? p2.min : p2.max;
  auto r = BoundSkewTree::Equal(h, 0) ? _unit_v_res : _unit_h_res;
  auto c = BoundSkewTree::Equal(h, 0) ? _unit_v_cap : _unit_h_cap;
  calcMergeDist(r, c, p1.val, delay1, p2.val, delay2, h + v, d1, d2);
  calcPtCoordOnLine(p1, p2, d1, d2, bal_pt);
  double incr_delay1 = 0;
  double incr_delay2 = 0;
  if (BoundSkewTree::Equal(h, 0)) {
    incr_delay1 = calcDelayIncrease(0, d1, p1.val, pattern);
    incr_delay2 = calcDelayIncrease(0, d2, p2.val, pattern);
  } else {
    incr_delay1 = calcDelayIncrease(d1, 0, p1.val, pattern);
    incr_delay2 = calcDelayIncrease(d2, 0, p2.val, pattern);
  }
  bal_pt.min = fmin(p1.min + incr_delay1, p2.min + incr_delay2);
  bal_pt.max = fmax(p1.max + incr_delay1, p2.max + incr_delay2);
}
__device__ double calcXBalPosition(const double& delay1, const double& delay2, const double& cap1, const double& cap2, const double& h,
                                       const double& v, const size_t& bal_ref_side){
  auto rc = _pattern == RCPattern::kHV ? _unit_v_res * _unit_h_cap : _unit_h_res * _unit_v_cap;
  // printf("%.12f %f %f %f %f\n",rc,delay1,delay2,h,v);
  double t = 0;
  if (bal_ref_side == kX) {
    // assume (x, v-y) and (h-x, y), then set y = 0
    t = delay2 - delay1 + _K[kH] * h * h - _K[kV] * v * v + _unit_h_res * h * cap2 - _unit_v_res * v * cap1;
  } else {
    // assume (x, y) and (h-x, v-y), then set y = 0
    t = delay2 - delay1 + _K[kH] * h * h + _K[kV] * v * v + cap2 * (_unit_h_res * h + _unit_v_res * v) + rc * h * v;
  }
  auto x = t / (_unit_h_res * (cap1 + cap2) + rc * v + 2 * h * _K[kH]);
  return x;
}
__device__ double calcYBalPosition(const double& delay1, const double& delay2, const double& cap1, const double& cap2, const double& h,
                                       const double& v, const size_t& bal_ref_side) {
  auto rc = _pattern == RCPattern::kHV ? _unit_v_res * _unit_h_cap : _unit_h_res * _unit_v_cap;
  double t = 0;
  auto r = _unit_v_res * (cap1 + cap2) + 2 * v * _K[kV] + rc * h;
  double y = 0;
  if (bal_ref_side == kX) {
    // assume (x, y) and (h-x, v-y), then set x = 0
    t = delay2 - delay1 + _K[kH] * h * h + _K[kV] * v * v + cap2 * (_unit_h_res * h + _unit_v_res * v) + rc * h * v;
    y = t / r;
    // LOG_FATAL_IF(y > v + kEpsilon) << "y: " << y << " is larger than v: " << v;
  } else {
    // assume (h-x, y) and (x, v-y), then set x = 0
    t = delay2 - delay1 + _K[kV] * v * v - _K[kH] * h * h + _unit_v_res * v * cap2 - _unit_h_res * h * cap1;
    y = t / r;
    // LOG_FATAL_IF(y < -kEpsilon) << "y: " << y << " is less than 0";
  }
  return y;
}
__device__ void calcBalPtNotOnLine(Pt& p1, Pt& p2, const size_t& timing_type, const size_t& bal_ref_side, double& d1, double& d2,
                                       Pt& bal_pt, const RCPattern& pattern) {
  auto h = fabs(p1.x - p2.x);
  auto v = fabs(p1.y - p2.y);
  // LOG_FATAL_IF(Equal(h, 0) || Equal(v, 0)) << "h or v is zero, which balance point is on line";
  // LOG_FATAL_IF(p1.x > p2.x) << "p1 is not left of p2";

  auto delay1 = timing_type == kMin ? p1.min : p1.max;
  auto delay2 = timing_type == kMin ? p2.min : p2.max;
  // printf("calcBalPtNotOnLine:%f,%f\n",p1.val, p2.val);//check
  auto x = calcXBalPosition(delay1, delay2, p1.val, p2.val, h, v, bal_ref_side);
  // printf("calcBalPtNotOnLine X:%f\n",x);
  double y = 0;
  if (x < 0) {
    y = bal_ref_side == kX ? calcYBalPosition(delay1, delay2, p1.val, p2.val, h, v, bal_ref_side) : -1;
    x = y >= 0 ? 0 : x;
  } else if (x > h) {
    y = bal_ref_side == kX ? v + 1 : calcYBalPosition(delay1, delay2, p1.val, p2.val, h, v, bal_ref_side);
    x = y <= v ? h : x;
  } else {
    y = bal_ref_side == kX ? v : 0;
  }
  // printf("calcBalPtNotOnLine:%f,%f,%f\n",x, y,h);
  if (x < 0) {
    // LOG_FATAL_IF(y >= 0) << "y is illegal";
    auto temp_pt = p1;
    auto incr_delay = calcDelayIncrease(h, v, p2.val, pattern);
    temp_pt.min = p2.min + incr_delay;
    temp_pt.max = p2.max + incr_delay;
    temp_pt.val = p2.val + _unit_h_cap * h + _unit_v_cap * v;
    calcBalPtOnLine(p1, temp_pt, timing_type, d1, d2, bal_pt, pattern);
    // LOG_FATAL_IF(d1 > kEpsilon) << "dist to p1 should be zero";
    auto new_incr_delay = calcDelayIncrease(0, d2, temp_pt.val, pattern);
    // LOG_FATAL_IF(!Equal(delay1, incr_delay + new_incr_delay + delay2)) << "delay is not equal";
    d2 += h + v;
  } else if (x > h) {
    // LOG_FATAL_IF(y <= v) << "y: " << y << " is not greater than v: " << v;
    auto temp_pt = p2;
    auto incr_delay = calcDelayIncrease(h, v, p1.val, pattern);
    temp_pt.min = p1.min + incr_delay;
    temp_pt.max = p1.max + incr_delay;
    temp_pt.val = p1.val + _unit_h_cap * h + _unit_v_cap * v;
    calcBalPtOnLine(temp_pt, p2, timing_type, d1, d2, bal_pt, pattern);
    // LOG_FATAL_IF(d2 > kEpsilon) << "dist to p2 should be zero";
    auto new_incr_delay = calcDelayIncrease(0, d1, temp_pt.val, pattern);
    // LOG_FATAL_IF(!Equal(delay2, incr_delay + new_incr_delay + delay1)) << "delay is not equal";
    d1 += h + v;
  } else {
    // LOG_FATAL_IF(y < -kEpsilon || y > v + kEpsilon) << "y: " << y << " is not in range [0, " << v << "]";
    bal_pt.x = p1.x + x;
    bal_pt.y = p1.y < p2.y ? p1.y + y : p1.y - y;
    auto incr_delay1 = calcDelayIncrease(x, y, p1.val, pattern);
    auto incr_delay2 = calcDelayIncrease(h - x, v - y, p2.val, pattern);
    bal_pt.min = fmin(p1.min + incr_delay1, p2.min + incr_delay2);
    bal_pt.max = fmax(p1.max + incr_delay1, p2.max + incr_delay2);
    d1 = x + y;
    d2 = h + v - d1;
    // LOG_FATAL_IF(!Equal(incr_delay1 + delay1, incr_delay2 + delay2)) << "delay is not equal";
  }
  // LOG_FATAL_IF(d1 + d2 < h + v - kEpsilon) << "dist out of range";
}
__device__ void calcBalBetweenPts(Pt& p1, Pt& p2, const size_t& timing_type, const size_t& bal_ref_side, double& d1, double& d2,
                                      Pt& bal_pt, const RCPattern& pattern){
  auto h = fabs(p1.x - p2.x);
  auto v = fabs(p1.y - p2.y);
  // printf("calcBalBetweenPts:%f,%f\n",h,v);
  if (BoundSkewTree::Equal(h, 0) || BoundSkewTree::Equal(v, 0)) {
    calcBalPtOnLine(p1, p2, timing_type, d1, d2, bal_pt, pattern);
  } else if (p1.x <= p2.x) {
    calcBalPtNotOnLine(p1, p2, timing_type, bal_ref_side, d1, d2, bal_pt, pattern);
  } else {
    calcBalPtNotOnLine(p2, p1, timing_type, bal_ref_side, d2, d1, bal_pt, pattern);
  }
}
__device__ void calcBalancePt(Area* cur,Side<Pts>& _bal_points){
  FOR_EACH_SIDE(end_side)
  {
    _bal_points[end_side].size=0;
  }
  if (BoundSkewTree::Equal(cur->radius, 0)) {
    return;
  }
  FOR_EACH_SIDE(end_side)
  {
    auto left_line = cur->lines[kLeft];
    auto right_line = cur->lines[kRight];
    auto left_pt = left_line.pts[end_side];
    auto right_pt = right_line.pts[end_side];
    left_pt.val = cur->get_left->cap;
    right_pt.val = cur->get_right->cap;
    auto bal_ref_side = (left_pt.x - right_pt.x) * (left_pt.y - right_pt.y) < 0 ? 1 - end_side : end_side;
    FOR_EACH_SIDE(timing_type)
    {
      double dist_to_left = 0, dist_to_right = 0;
      Pt bal_pt;
      // printf("left_pt:%f,%f,%f,right_pt:%f,%f,%f,%f\n",left_pt.x,left_pt.y,left_pt.min,right_pt.x,right_pt.y,right_pt.min,right_pt.max);//check
      calcBalBetweenPts(left_pt, right_pt, timing_type, bal_ref_side, dist_to_left, dist_to_right, bal_pt, _pattern);
      // printf("dist:%f,%f\n",dist_to_left,dist_to_right);
      // printf("left_pt:%f,%f,%f,right_pt:%f,%f,%f,%f\ndist_to_left:%f,%f\n",left_pt.x,left_pt.y,left_pt.min,right_pt.x,right_pt.y,right_pt.min,right_pt.max,dist_to_left,dist_to_right);
      if (!BoundSkewTree::Equal(dist_to_left, 0) && !BoundSkewTree::Equal(dist_to_right, 0)) {
        updatePtDelaysByEndSide(cur, end_side, bal_pt);
        // _bal_points[end_side].push_back(bal_pt);
        _bal_points[end_side].pts[_bal_points[end_side].size++]=bal_pt;
      }
    }
  }
  // printf("dist_to_left_size:%d,%d\n",_bal_points[0].size,_bal_points[1].size);
}
__device__ double calcSkewSlope(Area* cur){
  auto left_x = cur->lines[kLeft].pts[kHead].x;
  auto left_y = cur->lines[kLeft].pts[kHead].y;
  auto right_x = cur->lines[kRight].pts[kHead].x;
  auto right_y = cur->lines[kRight].pts[kHead].y;
  auto left_cap = cur->get_left->cap;
  auto right_cap = cur->get_right->cap;
  if (BoundSkewTree::Equal(left_x, right_x)) {
    return _unit_v_res * (left_cap + right_cap + cur->radius * _unit_v_cap);
  } else if (BoundSkewTree::Equal(left_y, right_y)) {
    return _unit_h_res * (left_cap + right_cap + cur->radius * _unit_h_cap);
  }
  // LOG_FATAL << "line is not horizontal or vertical";
}
__device__ void calcFmsBetweenPts(const Pt& high_skew_pt, const Pt& low_skew_pt, Pt& fms_pt){
  auto high_skew = ptSkew(high_skew_pt);
  auto low_skew = ptSkew(low_skew_pt);
  // LOG_FATAL_IF(low_skew > _skew_bound) << "low skew is larger than skew bound";
  // LOG_FATAL_IF(high_skew < low_skew + kEpsilon) << "high skew is less than low skew";
  auto dist = BoundSkewTree::distanceR(high_skew_pt, low_skew_pt);
  // LOG_FATAL_IF(dist <= kEpsilon) << "distance is less than epsilon";
  auto dist_to_low = dist * (_skew_bound - low_skew) / (high_skew - low_skew);
  calcPtCoordOnLine(high_skew_pt, low_skew_pt, dist - dist_to_low, dist_to_low, fms_pt);
}
__device__ bool calcFmsOnLine(Area* cur, Pt& pt, const Pt& q, const size_t& end_side,Side<Pts>& _bal_points,Side<Pts>& _fms_points){
  auto skew = ptSkew(pt);
  // printf("%f %f %f %f\n",pt.x,pt.y,pt.min,pt.max);
  if (BoundSkewTree::Equal(skew, _skew_bound) || skew < _skew_bound) {
    // _fms_points[end_side].push_back(pt);
    _fms_points[end_side].pts[_fms_points[end_side].size++]=pt;
    return true;
  }
  auto min_dist_pt = q;
  // std::ranges::for_each(_bal_points[end_side], [&min_dist_pt, &pt](const Pt& bal_pt) {
  for(int i=0;i<_bal_points[end_side].size;i++){
    if (BoundSkewTree::distanceR(pt, _bal_points[end_side].pts[i]) < BoundSkewTree::distanceR(pt, min_dist_pt)) {
      min_dist_pt = _bal_points[end_side].pts[i];
    }
  }
  skew = ptSkew(min_dist_pt);
  if (BoundSkewTree::Equal(skew, _skew_bound)) {
    // _fms_points[end_side].push_back(min_dist_pt);
    _fms_points[end_side].pts[_fms_points[end_side].size++]=min_dist_pt;
    return true;
  }
  if (skew < _skew_bound) {
    Pt fms_pt;
    calcFmsBetweenPts(pt, min_dist_pt, fms_pt);
    updatePtDelaysByEndSide(cur, end_side, fms_pt);
    if (!BoundSkewTree::Equal(ptSkew(fms_pt), _skew_bound)) {
      auto p1 = pt;
      auto p2 = min_dist_pt;
      updatePtDelaysByEndSide(cur, end_side, p1);
      updatePtDelaysByEndSide(cur, end_side, p2);
      // LOG_FATAL << "feasible merge section point should in skew bound";
    }
    _fms_points[end_side].pts[_fms_points[end_side].size++]=fms_pt;
    return true;
  }
  return false;
}
__device__ void calcFmsPt(Area* cur,Side<Pts>& _join_segment,Side<Pts>& _join_region,Side<Pts>& _bal_points,Side<Pts>& _fms_points,Side<Pt>&_join_corner){
  FOR_EACH_SIDE(end_side)
  {
    _fms_points[end_side].size=0;
    Side<Pt> candidate;
    if (end_side == kHead) {
      candidate[kLeft] = _join_region[kLeft].pts[0];
      candidate[kRight] = _join_region[kRight].pts[0];
    } else {
      int end1=_join_region[kLeft].size;
      int end2=_join_region[kRight].size;
      candidate[kLeft] = _join_region[kLeft].pts[end1-1];
      candidate[kRight] = _join_region[kRight].pts[end2-1];
    }
    bool exist = false;
    if (jrCornerExist(end_side,_join_segment)) {
      exist = calcFmsOnLine(cur, candidate[kLeft], _join_corner[end_side], end_side,_bal_points,_fms_points);
      if (exist) {
        exist = calcFmsOnLine(cur, candidate[kRight], _join_corner[end_side], end_side,_bal_points,_fms_points);
        if (!exist) {
          exist = calcFmsOnLine(cur, _join_corner[end_side], candidate[kLeft], end_side,_bal_points,_fms_points);
          // LOG_FATAL_IF(!exist) << "can't find feasible merge section on line";
        }
      } else {
        exist = calcFmsOnLine(cur, _join_corner[end_side], candidate[kRight], end_side,_bal_points,_fms_points);
        if (exist) {
          exist = calcFmsOnLine(cur, candidate[kRight], _join_corner[end_side], end_side,_bal_points,_fms_points);
          // LOG_FATAL_IF(!exist) << "can't find feasible merge section on line";
        }
      }
    } else {
      exist = calcFmsOnLine(cur, candidate[kLeft], candidate[kRight], end_side,_bal_points,_fms_points);
      if (exist) {
        calcFmsOnLine(cur, candidate[kRight], candidate[kLeft], end_side,_bal_points,_fms_points);
      }
    }
    uniquePtsLoc(_fms_points[end_side]);
  }
}
__device__ void mrBetweenJs(int id,Area* cur, const size_t& end_side,Side<Pts>&_join_segment,Side<Pts>&_bal_points,Side<Pts>&_fms_points,Side<Pt>&_join_corner){
  Pt mr_pts[32];
  int cnt=0;
  // printf("mrBetweenJs:%d\n",cur->mr.size);
  // std::ranges::for_each(_bal_points[end_side], [&mr_pts](const Pt& p) { mr_pts.push_back(p); });
  // std::ranges::for_each(_fms_points[end_side], [&mr_pts](const Pt& p) { mr_pts.push_back(p); });
  for(int i=0;i<_bal_points[end_side].size;i++){mr_pts[cnt++]=_bal_points[end_side].pts[i];}
  for(int i=0;i<_fms_points[end_side].size;i++){mr_pts[cnt++]=_fms_points[end_side].pts[i];}
  if (jrCornerExist(end_side,_join_segment) && ptSkew(_join_corner[end_side]) < _skew_bound + kEpsilon) {
    // mr_pts.push_back(_join_corner[end_side]);
    mr_pts[cnt++]=_join_corner[end_side];
  }
  if (cnt==0) {
    return;
  }
  auto left_line = cur->lines[kLeft];
  auto right_line = cur->lines[kRight];
  Pt ref_js_pt = end_side == kHead ? left_line.pts[end_side] : right_line.pts[end_side];
  // std::ranges::for_each(mr_pts, [&](Pt& pt) { pt.val = distance(pt, ref_js_pt); });
  for(int i=0;i<cnt;i++){mr_pts[i].val=BoundSkewTree::distanceR(mr_pts[i], ref_js_pt);}
  sortPtsByValDec(mr_pts,cnt);
  uniquePtsLoc(mr_pts,cnt);
  // if(id==0){
  //   printf("mrBetweenJs %d %d %d %d\n",cur->mr.size,_bal_points[end_side].size,_fms_points[end_side].size,cnt);
  // }
  for(int i=0;i<cnt;i++){cur->mr.pts[cur->mr.size++]=mr_pts[i];}
  // if(cur->mr.size>=10){
  //   for(int i=0;i<cur->mr.size;i++)printf("mrBetweenJs %f %f\n",cur->mr.pts[i].x,cur->mr.pts[i].y);
  // }
  // std::ranges::for_each(mr_pts, [&cur](const Pt& p) { cur->add_mr_point(p); });
}
__device__ bool jRisLine(Side<Pts>&_join_segment){
  auto left_head = _join_segment[kLeft].pts[kHead];
  auto left_tail = _join_segment[kLeft].pts[kTail];
  auto right_head = _join_segment[kRight].pts[kHead];
  auto right_tail = _join_segment[kRight].pts[kTail];
  auto min_x = fmin(fmin(left_head.x, left_tail.x),fmin(right_head.x, right_tail.x));
  auto min_y = fmin(fmin(left_head.y, left_tail.y),fmin(right_head.y, right_tail.y));
  auto max_x = fmax(fmin(left_head.x, left_tail.x),fmin(right_head.x, right_tail.x));
  auto max_y = fmax(fmin(left_head.y, left_tail.y),fmin(right_head.y, right_tail.y));
  if (BoundSkewTree::Equal(min_x, max_x) || BoundSkewTree::Equal(min_y, max_y)) {
    return true;
  }
  return false;
}
__device__ void fmsOfLineExist(Area* cur, const size_t& side, const size_t& idx,Side<Pts>&_join_segment,Side<Pts>&_join_region) {
  auto pt = _join_region[side].pts[idx];
  auto slope = calcSkewSlope(cur);
  auto dist = (ptSkew(pt) - _skew_bound) / slope;
  if (dist <= 0) {
    // cur->add_mr_point(pt);
    cur->mr.pts[cur->mr.size++]=pt;
  } else if (dist <= cur->radius) {
    auto relative_type = lineRelative(getJsLine(_join_segment,kLeft), getJsLine(_join_segment,kRight), side);
    calcRelativeCoord(pt, relative_type, dist);
    auto x = fabs(pt.x - _join_region[side].pts[idx].x);
    auto y = fabs(pt.y - _join_region[side].pts[idx].y);
    // LOG_FATAL_IF(!Equal(x, 0) && !Equal(y, 0)) << "not horizontal or vertical";
    auto incr_delay = side == kLeft ? calcDelayIncrease(x, y, cur->get_left->cap, _pattern)
                                    : calcDelayIncrease(x, y, cur->get_right->cap, _pattern);
    pt.min += incr_delay;
    pt.max = _skew_bound + pt.min;
    // cur->add_mr_point(pt);
    cur->mr.pts[cur->mr.size++]=pt;
  }
}
__device__ void mrOnJs(Area* cur, const size_t& side,Side<Pts>&_join_segment,Side<Pts>&_join_region,Side<Pts>&_fms_points){
  // LOG_FATAL_IF(_join_region[side].size() < 2) << "join region size is less than 2";
  // if(cur->x>46&&cur->x<47){
  //   for(int i=0;i<_join_segment[side].size;i++)printf("mrOnJs:%d %f %f\n",side,_join_segment[side].pts[i].x,_join_segment[side].pts[i].y);
  //   for(int i=0;i<_join_region[side].size;i++)printf("mrOnJs_jr:%d %f %f\n",side,_join_region[side].pts[i].x,_join_region[side].pts[i].y);
  //   for(int i=0;i<_fms_points[side].size;i++)printf("mrOnJs_fms:%d %f %f\n",side,_fms_points[side].pts[i].x,_fms_points[side].pts[i].y);
  // }
  auto other_side = side == kLeft ? kRight : kLeft;
  auto p = _join_region[side].pts[0];
  auto q = _join_region[other_side].pts[0];
  size_t jr_left_id = 1;
  if (_fms_points[kHead].size==0 && ptSkew(p) < ptSkew(q)) {
    for (; jr_left_id < _join_region[side].size - 1; ++jr_left_id) {
      if (BoundSkewTree::Equal(ptSkew(_join_region[side].pts[jr_left_id]), _skew_bound)) {
        break;
      }
    }
  }
  p = _join_region[side].pts[_join_region[side].size-1];
  q = _join_region[other_side].pts[_join_region[other_side].size-1];
  size_t jr_right_id = _join_region[side].size - 2;
  if (_fms_points[kTail].size==0 && ptSkew(p) < ptSkew(q)) {
    for (; jr_right_id >= jr_left_id; --jr_right_id) {
      if (BoundSkewTree::Equal(ptSkew(_join_region[side].pts[jr_right_id]), _skew_bound)) {
        break;
      }
    }
  }
  if (side == kLeft) {
    for (size_t i = jr_left_id; i <= jr_right_id; ++i) {
      fmsOfLineExist(cur, side, i,_join_segment,_join_region);
    }
  } else {
    for (size_t i = jr_right_id; i >= jr_left_id; --i) {
      fmsOfLineExist(cur, side, i,_join_segment,_join_region);
    }
  }
}
__device__ void constructFeasibleMr(int id,Area* parent, Area* left, Area* right,Side<Pts>&_join_segment,Side<Pts>&_join_region,Side<Pts>&_bal_points,Side<Pts>&_fms_points,Side<Pt>&_join_corner){
  if (calcAreaLineType(parent) == LineType::kManhattan) {
    // parallel manhattan arc
    mrBetweenJs(id,parent, kHead,_join_segment,_bal_points,_fms_points,_join_corner);
    if (!((parent->mr.size)==0) && !jRisLine(_join_segment)) {
      mrBetweenJs(id,parent, kTail,_join_segment,_bal_points,_fms_points,_join_corner);
    }
  } else {
    // parallel horizontal or vertical arc
    mrBetweenJs(id,parent, kHead,_join_segment,_bal_points,_fms_points,_join_corner);
    mrOnJs(parent, kLeft,_join_segment,_join_region,_fms_points);
    mrBetweenJs(id,parent, kTail,_join_segment,_bal_points,_fms_points,_join_corner);
    // printf("constructFeasibleMr here\n");
    if (parent->radius > kEpsilon) {
      mrOnJs(parent, kRight,_join_segment,_join_region,_fms_points);
    }
  }
}
__device__ bool existFmsOnJr(Side<Pts>&_join_region,Side<Pts>&_fms_points) {
  if (!(_fms_points[kHead].size==0) || !(_fms_points[kTail].size==0)) {
    return true;
  }
  FOR_EACH_SIDE(side)
  {
    // for (auto pt : _join_region[side]) {
    for(int i=0;i<_join_region[side].size;i++){
      if (ptSkew(_join_region[side].pts[i]) <= _skew_bound) {
        return true;
      }
    }
  }
  return false;
}
__device__ void calcMinSkewSection(Area* cur,Side<Pts>&_join_region) {
  auto min_skew = DOUBLE_MAX;
  auto min_skew_side = kLeft;
  FOR_EACH_SIDE(side)
  {
    auto min_side_skew = DOUBLE_MAX;
    // std::ranges::for_each(_join_region[side], [&](const Pt& pt) { min_side_skew = std::min(min_side_skew, ptSkew(pt)); });
    for(int i=0;i<_join_region[side].size;i++){min_side_skew = fmin(min_side_skew, ptSkew(_join_region[side].pts[i]));}
    if (min_side_skew < min_skew) {
      min_skew = min_side_skew;
      min_skew_side = side;
    }
  }
  // std::ranges::for_each(_join_region[min_skew_side], [&](const Pt& pt) {
  for(int i=0;i<_join_region[min_skew_side].size;i++){
    if (BoundSkewTree::Equal(ptSkew(_join_region[min_skew_side].pts[i]), min_skew)) {
      // cur->add_mr_point(pt);
      cur->mr.pts[cur->mr.size++]=_join_region[min_skew_side].pts[i];
    }
  }
}
__device__ void calcDetourEdgeLen(Area* cur) {
  auto left_pt = cur->lines[kLeft].pts[kHead];
  auto right_pt = cur->lines[kRight].pts[kHead];
  left_pt.val = cur->get_left->cap;
  right_pt.val = cur->get_right->cap;
  auto delta = ptSkew(cur->mr.pts[0]) - _skew_bound;
  // LOG_FATAL_IF(delta <= 0) << "remain skew less than 0";
  auto h = fabs(left_pt.x - right_pt.x);
  auto v = fabs(left_pt.y - right_pt.y);
  if (left_pt.max > right_pt.max) {
    right_pt.max = left_pt.max - delta - calcDelayIncrease(h, v, right_pt.val, _pattern);
    double d1 = 0, d2 = 0;
    Pt bal_pt;
    calcBalBetweenPts(left_pt, right_pt, kMax, kX, d1, d2, bal_pt, _pattern);
    // LOG_FATAL_IF(d1 > kEpsilon) << "dist to left_pt should be zero";
    cur->_edge_len[kLeft]= 0;
    cur->_edge_len[kRight]= d2;
  } else {
    left_pt.max = right_pt.max - delta - calcDelayIncrease(h, v, left_pt.val, _pattern);
    double d1 = 0, d2 = 0;
    Pt bal_pt;
    calcBalBetweenPts(left_pt, right_pt, kMax, kX, d1, d2, bal_pt, _pattern);
    // LOG_FATAL_IF(d2 > kEpsilon) << "dist to right_pt should be zero";
    cur->_edge_len[kLeft]=d1;
    cur->_edge_len[kRight]= 0;
  }
}
__device__ void refineMrDelay(Area* cur){
  // auto mr = cur->mr;
  // std::ranges::for_each(mr, [&](Pt& pt) { pt.min = pt.max - _skew_bound; });
  for(int i=0;i<cur->mr.size;i++){cur->mr.pts[i].min=cur->mr.pts[i].max-_skew_bound;}
  // cur->mr=mr;
}
__device__ void constructInfeasibleMr(Area* parent, Area* left, Area* right,Side<Pts>&_join_region) 
{
  calcMinSkewSection(parent,_join_region);
  calcDetourEdgeLen(parent);
  refineMrDelay(parent);
}
__device__ void constructTrrMr(Area* cur,Side<Trr>&_ms){
  Trr trr_left;
  buildTrr(_ms[kLeft], cur->_edge_len[kLeft], trr_left);
  Trr trr_right;
  buildTrr(_ms[kRight], cur->_edge_len[kRight], trr_right);

  Trr intersect;
  BoundSkewTree::makeIntersect(trr_left, trr_right, intersect);
  BoundSkewTree::trrCore(intersect, intersect);
  Pt mr[32];
  int cnt=0;
  BoundSkewTree::trrToRegion(intersect, mr,cnt);//bug
  // printf("trrToRegion over :%d\n",cnt);
  auto old_pt = cur->mr.pts[0];
  // std::ranges::for_each(mr, [&](Pt& p) {
  for(int i=0;i<cnt;i++){
    mr[i].min = old_pt.min;
    mr[i].max = old_pt.max;
  }
  // cur->set_mr(mr);
  for(int i=0;i<cnt;i++)cur->mr.pts[i]=mr[i];
  cur->mr.size=cnt;
}
__device__ bool comparePts(const Pt& p1, const Pt& p2, float kEpsilon) {
    if (p1.x + kEpsilon < p2.x) return true;
    if (p2.x + kEpsilon < p1.x) return false;
    return p1.y < p2.y;
}
// 非递归的奇偶排序实现，适合GPU设备
__device__ void deviceSort(Pt* pts, int size) {
    bool sorted = false;
    while (!sorted) {
        sorted = true;
        
        // 奇偶排序的奇阶段
        for (int i = 1; i < size - 1; i += 2) {
            if (comparePts(pts[i+1], pts[i], kEpsilon)) {
                Pt temp = pts[i];
                pts[i] = pts[i+1];
                pts[i+1] = temp;
                sorted = false;
            }
        }
        
        // 奇偶排序的偶阶段
        for (int i = 0; i < size - 1; i += 2) {
            if (comparePts(pts[i+1], pts[i], kEpsilon)) {
                Pt temp = pts[i];
                pts[i] = pts[i+1];
                pts[i+1] = temp;
                sorted = false;
            }
        }
    }
}
__device__ void convexHull(Pt* pts,int size,int &cnt){
  // check pts num
  if (size < 2) {
    cnt=size;
    return;
  }
  if (size == 2) {
    cnt=2;
    auto dist = BoundSkewTree::distanceR(pts[0], pts[1]);
    if (BoundSkewTree::Equal(dist, 0)) {
      cnt=1;
    }
    return;
  }
  // printf("convexHull:%f\n",pts.pts[0].x);
  // calculate convex hull by Andrew algorithm
  // std::ranges::sort(pts, [](const Pt& p1, const Pt& p2) { return p1.x + kEpsilon < p2.x || (Equal(p1.x, p2.x) && p1.y < p2.y); });
  deviceSort(pts, size);
  // std::vector<Pt> ans(2 * pts.size());
  Pt ans[32]; 
  size_t k = 0;
  for (size_t i = 0; i < size; ++i) {
    while (k > 1 && BoundSkewTree::crossProductR(ans[k - 2], ans[k - 1], pts[i]) <= kEpsilon) {
      --k;
    }
    ans[k++] = pts[i];
  }
  for (size_t i = size - 1, t = k + 1; i > 0; --i) {
    while (k >= t && BoundSkewTree::crossProductR(ans[k - 2], ans[k - 1], pts[i - 1]) <= kEpsilon) {
      --k;
    }
    ans[k++] = pts[i - 1];
  }
  // pts = {ans.begin(), ans.begin() + k - 1};
  for (size_t i = 0; i < k - 1; ++i){ pts[i] = ans[i];}
  cnt=k-1;
}
__device__ void calcConvexHull(Area* cur) {
  Pt pts[32];
  int cnt=0;
  // auto mr = cur->mr;
  // printf("calcConvexHull%d,%f,%f,%f,%f\n",cur->mr.size,cur->mr.pts[0].x,cur->mr.pts[0].y,cur->mr.pts[1].x,cur->mr.pts[1].y);
  for(int i=0;i<cur->mr.size;i++)pts[i]=cur->mr.pts[i];
  convexHull(pts,cur->mr.size,cnt);
  // printf("calcConvexHull:mr:%d\n",cur->mr.size);
  // cur->set_convex_hull(mr);
  // cur->convex_hull=mr;
  cur->convex_hull.size=cnt;
  for(int i=0;i<cnt;i++)cur->convex_hull.pts[i]=pts[i];
  // printf("calcConvexHull1%d,%f,%f,%f,%f\n",cur->convex_hull.size,cur->convex_hull.pts[0].x,cur->convex_hull.pts[0].y);
}
__device__ void constructMr(int id,int beginID,Area* parent, Area* left, Area* right,Side<Pts>& join_segment,Side<Pts>& join_region,Side<Pts>& bal_points,Side<Pts>& fms_points,Side<Pt>&join_corner,Side<Trr>& _ms)
{
  // printf("constructMr begin\n");
  calcJr(id,beginID,parent, left, right,join_segment,join_region);
  // if(id+beginID==34)for(int i=0;i<join_region[1].size;i++)printf("mrOnJs_jr:%d %f %f\n",1,join_region[1].pts[i].x,join_region[1].pts[i].y);
  // if(id+beginID==12){
  //   for(int i=0;i<join_segment[0].size;i++)printf("jm0 %f %f\n",join_segment[0].pts[i].x,join_segment[0].pts[i].y);
  //   for(int i=0;i<join_segment[1].size;i++)printf("jm1 %f %f\n",join_segment[1].pts[i].x,join_segment[1].pts[i].y);
  // }
  // printf("calcJr over\n");
  // printf("parent->lines[0].pts[0].x:%f\n",parent->lines[0].pts[0].x);
  calcJrCorner(parent,join_segment,join_corner);
  // printf("calcJrCorner over\n");
  calcBalancePt(parent,bal_points);
  // printf("bal_points:%d,%d\n",bal_points[1].size,bal_points[0].size);
  // printf("calcBalancePt over\n");
  calcFmsPt(parent,join_segment,join_region,bal_points,fms_points,join_corner);
  // printf("calcFmsPt over\n");
  if (existFmsOnJr(join_region,fms_points)) {
    // printf("constructMr:%d,%d:",bal_points[0].size,bal_points[1].size);
    constructFeasibleMr(id,parent, left, right,join_segment,join_region,bal_points,fms_points,join_corner);
  } else {
    constructInfeasibleMr(parent, left, right,join_region);
  }
  if (BoundSkewTree::lineType(parent->lines[kLeft]) == LineType::kManhattan && parent->_edge_len[kLeft] >= 0) {
    // LOG_FATAL_IF(parent->get_edge_len(kRight) < 0) << "right edge length is negative";
    constructTrrMr(parent,_ms);
  }
  // auto mr = parent->mr;
  // printf("constructMr:%d\n",parent->mr.size);
  uniquePtsLoc(parent->mr); //bug
  // parent->set_mr(mr);
  calcConvexHull(parent);
}
__global__ void merge(Area* areas,int beginID,int areasize,Side<Pts>* join_segment, Side<Pts>* join_region,Side<Pts>* bal_points,Side<Pts>* fms_points,Side<Trr>* ms) {
  int id = blockIdx.x * blockDim.x + threadIdx.x; 
  if(id>=areasize)return;
  // printf("_K:%.12f,%.12f\n",_K[0],_K[1]);
  // printf("_unit_h_cap:%f\n",_unit_h_cap);
  // printf("l,r:%d %d\n",areas[id+beginID].l,areas[id+beginID].r);
  int size=areas[id+beginID].r-areas[id+beginID].l+1;
  // printf("mergesize:%d,%d\n",id+beginID,size);
  if(size<=1)return;
  // printf("l,r:%d %d %d %d\n",id+beginID,areas[id+beginID].left,areas[id+beginID].l,areas[id+beginID].r);
  // parent->set_left(left);
  // parent->set_right(right);
  // left->set_parent(parent);
  // right->set_parent(parent);
  Side<Pt> join_corner;
  Area* parent=&areas[id+beginID];
  Area* left=&areas[areas[id+beginID].left];
  Area* right=&areas[areas[id+beginID].right];
  parent->get_left=left;
  parent->get_right=right;
  // printf("%f\n",left->cap);
  // calcJS(parent, left, right,join_segment[id],join_region[id],ms[id]);
  // printf("\njoin_segment_size:%d,%d,%f\n",join_segment[id][0].size,join_segment[id][1].size,join_segment[id][0].pts[0].x);
  // printf("calcJS over\n");
  parent->_edge_len[kLeft]=-1;
  parent->_edge_len[kRight]=-1;
  
  auto dist = parent->radius;
  jsProcess(parent,join_segment[id],join_region[id]);//bug
  // printf("%f\n",parent->lines[0].pts[0].x);
  // printf("jsProcess over\n");
  // auto left_line = parent->get_line(kLeft);
  // auto right_line = parent->get_line(kRight);
  auto left_line =parent->lines[kLeft];
  auto right_line = parent->lines[kRight];
  constructMr(id,beginID,parent, left, right,join_segment[id],join_region[id],bal_points[id],fms_points[id],join_corner,ms[id]);
  // printf("constructMr over\n");
  if (BoundSkewTree::lineType(getJsLine(join_segment[id],kLeft)) == LineType::kManhattan) {
    // LOG_FATAL_IF(lineType(getJsLine(kRight)) != LineType::kManhattan) << "right js is not manhattan";
    if (BoundSkewTree::isSegmentTrr(ms[id][kLeft])) {
      BoundSkewTree::msToLine(ms[id][kLeft], left_line);
    }
    if (BoundSkewTree::isSegmentTrr(ms[id][kRight])) {
      BoundSkewTree::msToLine(ms[id][kRight], right_line);
    }
  }
  parent->lines[kLeft]=left_line;
  parent->lines[kRight]=right_line;

  parent->radius=dist;
  if (parent->_edge_len[kLeft] + parent->_edge_len[kRight] < 0) {
    parent->cap=(left->cap + right->cap + parent->radius * _unit_h_cap);
  } else {
    parent->cap=(left->cap + right->cap
                         + (parent->_edge_len[kLeft] + parent->_edge_len[kRight]) * _unit_h_cap);
  }
  // printf("%d %f,%f %d\n",id+beginID,parent->radius,parent->cap,parent->mr.size);
  // if(id+beginID==34||id+beginID==36||id+beginID==35)for(int i=0;i<parent->mr.size;i++)printf("%d %f %f\n",id,parent->mr.pts[i].x,parent->mr.pts[i].y);
}
__global__ void initSidejoin(Side<Pts>* join,Pt* ptsl,Pt* ptsr,int c,int num){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= num) return;
  join[id][0].size=0;
  join[id][1].size=0;
  join[id][0].pts=&ptsl[c*id];
  join[id][1].pts=&ptsr[c*id];
}
void BoundSkewTree::initSidePts(Side<Pts>* &join,Pt*&ptsl,Pt*&ptsr,int num,int c){
  CUDA_CHECK(cudaMalloc(&ptsl, c*num*sizeof(Pt)));
  CUDA_CHECK(cudaMalloc(&ptsr, c*num*sizeof(Pt)));
  CUDA_CHECK(cudaMalloc(&join, num*sizeof(Side<Pts>)));
  cudaMemset(ptsl, 0, c*num*sizeof(Pt));
  cudaMemset(ptsr, 0, c*num*sizeof(Pt));
  cudaMemset(join, 0, num*sizeof(Side<Pts>));
  // std::cout<<8*sizeof(Pt)<<' '<<sizeof(Side<Pts>)<<' '<<sizeof(Side<Trr>)<<'\n';
  int blockSize = 256;
  int gridSize = (num + blockSize - 1) / blockSize;
  initSidejoin<<<gridSize, blockSize>>>(join,ptsl,ptsr,c,num);
  cudaDeviceSynchronize();
}
void BoundSkewTree::initms(Side<Trr>*& ms,int num){
  cudaMalloc(&ms, num* sizeof(Side<Trr>));
  cudaMemset(ms, 0, num* sizeof(Side<Trr>));
}
// Inst* BoundSkewTreecpu::genBufInst(const std::string& prefix, const Point& location)
// {
//   auto buf_name = prefix + "_buf";
//   auto buf_inst = new Inst(buf_name, location, InstType::kBuffer);
//   return buf_inst;
// }
__global__ void noneInputTopologyConvert(Area* areas,int pinsize,int end){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int size=end-pinsize;
  if(id>=size)return ;
  id+=pinsize;
  if(areas[id].p==-1){
    areas[id].snake=0;
    return;
  }
  int side;
  // printf("%d,%d,%d,%d\n",areas[id].p,areas[areas[id].p].left,areas[areas[id].p].right,id);
  if(areas[areas[id].p].left==id)side=0;
  else side=1;
  auto edge_len = areas[areas[id].p]._edge_len[side];
  auto snake = edge_len - BoundSkewTree::distanceR(areas[areas[id].p].location, areas[id].location);
  // printf("%f,%f,edge_len:%f,%f,%f\n",areas[id].location.x,areas[id].location.y,areas[areas[id].p]._edge_len[0],areas[areas[id].p]._edge_len[1],snake);
  areas[id].snake=snake;
  
}
void BoundSkewTree::convert(){
  int blockSize = 256;
  int gridSize = (levelsum[maxlevel]-pinsize + blockSize - 1) / blockSize;
  noneInputTopologyConvert<<<gridSize, blockSize>>>(areas,pinsize,levelsum[maxlevel]-1);
  cudaDeviceSynchronize();
}
void BoundSkewTree::recursiveBottomUp(){
  size_t h_kHead = 0;
  size_t h_kTail = 1;
  size_t h_kLeft = 0;
  size_t h_kRight = 1;
  size_t h_kMin = 0;
  size_t h_kMax = 1;
  size_t h_kX = 0;
  size_t h_kY = 1;
  size_t h_kH = 0;
  size_t h_kV = 1;

  // int db_unit=1000;
  // double unit_h_cap=1.94215e-06;
  // double unit_h_res=0.00178571;
  // double unit_v_cap=2.90402e-06;
  // double unit_v_res=0.00075;
  int db_unit = Timing::_db_unit;
  double unit_h_cap=Timing::_unit_h_cap;
  double unit_h_res=Timing::_unit_h_res;
  double unit_v_cap=Timing::_unit_v_cap;
  double unit_v_res=Timing::_unit_v_res;
  double skew_bound=g_skew_bound;
  double kepsilon=1e-7;
  RCPattern pattern = RCPattern::kHV;
  Side<double> K(0.5 * unit_h_res * unit_h_cap, 0.5 * unit_v_res* unit_v_cap);
  double _DOUBLE_MAX=std::numeric_limits<double>::max();
  double _DOUBLE_MIN=std::numeric_limits<double>::min();
  cudaMemcpyToSymbol(kHead, &h_kHead, sizeof(size_t));
  cudaMemcpyToSymbol(kTail, &h_kTail, sizeof(size_t));
  cudaMemcpyToSymbol(kLeft, &h_kLeft, sizeof(size_t));
  cudaMemcpyToSymbol(kRight, &h_kRight, sizeof(size_t));
  cudaMemcpyToSymbol(kMin, &h_kMin, sizeof(size_t));
  cudaMemcpyToSymbol(kMax, &h_kMax, sizeof(size_t));
  cudaMemcpyToSymbol(kX, &h_kX, sizeof(size_t));
  cudaMemcpyToSymbol(kY, &h_kY, sizeof(size_t));
  cudaMemcpyToSymbol(kH, &h_kH, sizeof(size_t));
  cudaMemcpyToSymbol(kV, &h_kV, sizeof(size_t));
  cudaMemcpyToSymbol(_K, &K, sizeof(Side<double>));
  cudaMemcpyToSymbol(_pattern, &pattern, sizeof(RCPattern));
  cudaMemcpyToSymbol(DOUBLE_MAX, &_DOUBLE_MAX, sizeof(double));
  cudaMemcpyToSymbol(DOUBLE_MIN, &_DOUBLE_MIN, sizeof(double));
  cudaMemcpyToSymbol(_db_unit, &db_unit, sizeof(int));
  cudaMemcpyToSymbol(_unit_h_cap, &unit_h_cap, sizeof(double));
  cudaMemcpyToSymbol(_unit_h_res, &unit_h_res, sizeof(double));
  cudaMemcpyToSymbol(_unit_v_cap, &unit_v_cap, sizeof(double));
  cudaMemcpyToSymbol(_unit_v_res, &unit_v_res, sizeof(double));
  cudaMemcpyToSymbol(_skew_bound, &skew_bound, sizeof(double));
  cudaMemcpyToSymbol(kEpsilon, &kepsilon, sizeof(double));
  // std::cout<<"recursiveBottomUp\n";

  Side<Pts>* join_segment;
  Side<Pts>* join_region;
  Side<Pts>* bal_points;
  Side<Trr>* ms;
  Side<Pts>* fms_points;
  Pt* pts[8];
  // std::cout<<pinsize<<'\n';
  initSidePts(join_segment,pts[0],pts[1],pinsize,16);
  initSidePts(join_region,pts[2],pts[3],pinsize,16);
  initSidePts(bal_points,pts[4],pts[5],pinsize,16);
  initSidePts(fms_points,pts[6],pts[7],pinsize,16);
  initms(ms,pinsize);
  for(int i=maxlevel;i>0;i--){
    // if(i==7)break;
    int blockSize = 256;
    int size=levelsum[i]-levelsum[i-1];
    int gridSize = (size + blockSize - 1) / blockSize;
    // if(levelsum[i-1]<1148)break;
    // std::cout<<"level:"<<i<<','<<levelsum[i-1]<<','<<size<<'\n';
    calcJS<<<gridSize, blockSize>>>(areas,levelsum[i-1],size,join_segment,join_region,ms);
    merge<<<gridSize, blockSize>>>(areas,levelsum[i-1],size,join_segment,join_region,bal_points,fms_points,ms);

    // std::cout<<"merge over\n";
    // cudaDeviceReset();
  }
  cudaFree(join_segment);
  cudaFree(join_region);
  cudaFree(bal_points);
  cudaFree(ms);
  cudaFree(fms_points);
  for(int i=0;i<8;i++)cudaFree(pts[i]);
}

}
