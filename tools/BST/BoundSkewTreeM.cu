#include <cub/cub.cuh>
#include "BoundSkewTree.h"
namespace gcts {
__device__ RCPattern _patternM;
__device__ double DOUBLE_MAXM;
__device__ double DOUBLE_MINM;
__device__  double _unit_h_capM;
__device__  double _unit_h_resM;
__device__  double _unit_v_capM;
__device__  double _unit_v_resM;
__device__  double _skew_boundM;
__device__  double kEpsilonM;  
__device__ bool isTrrContain(const Trr& small, const Trr& huge)
{
  if (small._x_high <= huge._x_high + kEpsilonM && small._x_low >= huge._x_low - kEpsilonM && small._y_high <= huge._y_high + kEpsilonM
      && small._y_low >= huge._y_low - kEpsilonM) {
    return true;
  }
  return false;
}
__device__ bool EqualM(const double& a, const double& b){
  return fabs(a - b) < 1e-7;
}
__device__ void checkMsM(Trr& ms){
  auto x_low = ms._x_low;
  auto x_high = ms._x_high;
  if (EqualM(x_low, x_high)) {
    auto avg = (x_low + x_high) / 2;
    ms._x_low=avg;
    ms._x_high=avg;
  }
  auto y_low = ms._y_low;
  auto y_high = ms._y_high;
  if (EqualM(y_low, y_high)) {
    auto avg = (y_low + y_high) / 2;
    ms._y_low=avg;
    ms._y_high=avg;
  }
  // LOG_FATAL_IF(ms.x_low() > ms.x_high() || ms.y_low() > ms.y_high()) << "ms is not valid";
}
__device__ void makeIntersectM(Trr& ms1, Trr& ms2, Trr& intersect){
  intersect._x_low=fmax(ms1._x_low, ms2._x_low);
  intersect._x_high=fmin(ms1._x_high, ms2._x_high);
  intersect._y_low=fmax(ms1._y_low, ms2._y_low);
  intersect._y_high=fmin(ms1._y_high, ms2._y_high);
  checkMsM(intersect);
}
__device__ LineType lineTypeM(const Pt& p1, const Pt& p2){
  auto d_x = fabs(p1.x - p2.x);
  auto d_y = fabs(p1.y - p2.y);
  if (EqualM(d_x, d_y)) {
    return LineType::kManhattan;
  } else if (EqualM(d_x, 0)) {
    return LineType::kVertical;
  } else if (EqualM(d_y, 0)) {
    return LineType::kHorizontal;
  } else if (d_x > d_y) {
    return LineType::kFlat;
  } else {
    return LineType::kTilt;
  }
}
__device__ LineType lineTypeM(const Line& l){
  size_t kHead = 0;
  size_t kTail = 1;
  return lineTypeM(l.pts[kHead], l.pts[kTail]);
}
__device__ double distanceM(const Pt& p1, const Pt& p2){
  return fabs(p1.x - p2.x) + fabs(p1.y - p2.y);
}
__device__ bool isSameM(const Pt& p1, const Pt& p2){
  auto dist = distanceM(p1, p2);
  return EqualM(dist, 0);
}
__device__ bool onLineM(Pt& p, const Line& l){
  size_t kHead = 0;
  size_t kTail = 1;
  double kEpsilon = 1e-7;
  auto len = distanceM(l.pts[kHead], l.pts[kTail]);
  auto len_to_head = distanceM(p, l.pts[kHead]);
  auto len_to_tail = distanceM(p, l.pts[kTail]);
  if (fabs(len_to_head + len_to_tail - len) < 2 * kEpsilon) {
    if (EqualM(len_to_head, 0)) {
      p = l.pts[kHead];
      return true;
    } else if (EqualM(len_to_tail, 0)) {
      p = l.pts[kTail];
      return true;
    } else {
      auto delta_x = fabs(l.pts[kTail].x - l.pts[kHead].x);
      auto delta_y = fabs(l.pts[kTail].y - l.pts[kHead].y);
      if (delta_y > delta_x) {
        auto temp_x = (l.pts[kTail].x - l.pts[kHead].x) * (p.y - l.pts[kHead].y) / (l.pts[kTail].y - l.pts[kHead].y) + l.pts[kHead].x;
        if (EqualM(temp_x, p.x)) {
          p.x = temp_x;
          return true;
        }
      } else {
        auto temp_y = (l.pts[kTail].y - l.pts[kHead].y) * (p.x - l.pts[kHead].x) / (l.pts[kTail].x - l.pts[kHead].x) + l.pts[kHead].y;
        if (EqualM(temp_y, p.y)) {
          p.y = temp_y;
          return true;
        }
      }
    }
  }
  return false;
}
__device__ bool isManhattanArea(Area* cur) {
  size_t kHead = 0;
  size_t kTail = 1;
  // auto mr = cur->get_mr();
  if (cur->mr.size == 1) {
    return true;
  }
  if (cur->mr.size == 2 && lineTypeM(cur->mr.pts[kHead], cur->mr.pts[kTail]) == LineType::kManhattan) {
    return true;
  }
  return false;
}
__device__ bool isRegionContain(const Pt& p, const Region& region){
  auto is_in_region = false;
  auto p_x = p.x;
  auto p_y = p.y;
  auto n = region.size;
  auto j = n - 1;
  for (size_t i = 0; i < n; j = i, ++i) {
    auto s_x = region.pts[i].x;
    auto s_y = region.pts[i].y;
    auto t_x = region.pts[j].x;
    auto t_y = region.pts[j].y;
    if ((s_y < p_y && t_y >= p_y) || (t_y < p_y && s_y >= p_y)) {
      if (s_x + (p_y - s_y) / (t_y - s_y) * (t_x - s_x) < p_x) {
        is_in_region = !is_in_region;
      }
    }
  }
  if (is_in_region) {
    return true;
  }
  auto pt = p;
  for (size_t i = 0; i < region.size; ++i) {
    auto j = (i + 1) % region.size;
    if (onLineM(pt, Line(region.pts[i], region.pts[j]))) {
      return true;
    }
  }
  return false;
}
__device__ void getMrLines(Line* lines,Region mr){
  for (size_t i = 0; i < mr.size; ++i) {
      auto j = (i + 1) % mr.size;
      lines[i]=Line(mr.pts[i],mr.pts[j]);
    }
}
__device__ double msDistanceM(Trr& ms1, Trr& ms2){
  checkMsM(ms1);
  checkMsM(ms2);
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
__device__ double ptToTrrDist(Pt& p, Trr& ms,double DOUBLE_MAX){
  Trr pt_trr(p, 0);
  if (isTrrContain(pt_trr, ms)) {
    return 0;
  }
  // std::vector<Trr> trrs(4, ms);
  Trr trrs[4];
  for(int i=0;i<4;i++)trrs[i]=ms;
  size_t kHead = 0;
  size_t kTail = 1;
  size_t kLeft = 0;
  size_t kRight = 1;
  trrs[kLeft + kHead]._y_low=ms._y_high;
  trrs[kLeft + kTail]._y_high=ms._y_low;
  trrs[kRight + kHead]._x_low=ms._x_high;
  trrs[kRight + kTail]._x_high=ms._x_low;
  double min_dist = DOUBLE_MAX;
  // std::ranges::for_each(trrs, [&pt_trr, &min_dist](auto& trr) {
  for(int i=0;i<4;i++){
    double dist = msDistanceM(pt_trr, trrs[i]);
    min_dist = fmin(min_dist, dist);
  }
  return min_dist;
}
__device__ bool isTrrArea(Area* cur) {
    size_t kHead = 0;
  size_t kTail = 1;
  if (isManhattanArea(cur)) {
    return true;
  }
  // auto mr = cur->get_mr();
  if (cur->mr.size!= 4) {
    return false;
  }
  int count = 0;
  Line mrLines[8];
  // for (auto line : cur->getMrLines()) {
  getMrLines(mrLines,cur->mr);
  for(int i=0;i<cur->mr.size;i++){
    if (lineTypeM(mrLines[i]) == LineType::kManhattan) {
      count++;
    }
    auto t_min = std::abs(mrLines[i].pts[kHead].min - mrLines[i].pts[kTail].min);
    auto t_max = std::abs(mrLines[i].pts[kHead].max - mrLines[i].pts[kTail].max);
    if (t_min > kEpsilon || t_max > kEpsilon) {
      return false;
    }
  }
  return count == 2;
}
__device__ void calcBsLocatedM(Area* cur, Pt& pt, Line& result_line) {
    // 1. 直接遍历mr.pts，避免填充临时数组
    for (int i = 0; i < cur->mr.size; ++i) {
        int j = (i + 1) % cur->mr.size;
        Line line(cur->mr.pts[i], cur->mr.pts[j]);
        
        // 2. 提前终止条件
        if (onLineM(pt, line)) {
            result_line = line;
            return; // 找到匹配线段
        }
    }
}
__device__ void makeDiamond(Trr& trr,const Pt& point, const double& radius)
  {
    auto val = point.x - point.y;
    trr._x_low = val - radius;
    trr._x_high = val + radius;
    val = point.x + point.y;
    trr._y_low = val - radius;
    trr._y_high = val + radius;
  }
__device__ bool is_empty(Trr& trr) 
  {
    auto x_interval = Interval(trr._x_low, trr._x_high);
    auto y_interval = Interval(trr._y_low, trr._y_high);
    return x_interval.is_empty() || y_interval.is_empty();
  }
__device__ void enclose(Trr& trr,const Trr& other){
    if (is_empty(trr)) {
      trr._x_low = other._x_low;
      trr._x_high = other._x_high;
      trr._y_low = other._y_low;
      trr._y_high = other._y_high;
    } else {
      trr._x_low = fmin(trr._x_low, other._x_low);
      trr._x_high = fmax(trr._x_high, other._x_high);
      trr._y_low = fmin(trr._y_low, other._y_low);
      trr._y_high = fmax(trr._y_high, other._y_high);
    }
  }
__device__ void lineToMsM(Trr& ms, const Pt& p1, const Pt& p2)
{
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
  checkMsM(ms);
}
__device__ void lineToMsM(Trr& ms, const Line& l){
  size_t kHead = 0;
  size_t kTail = 1;
  lineToMsM(ms, l.pts[kHead], l.pts[kTail]);
}
__device__ void mrToTrr(const Region& mr, Trr& trr) 
{
  size_t kHead = 0;
  size_t kTail = 1;
  if (mr.size == 1) {
    trr.makeDiamond(mr.pts[0], 0);
    return;
  }
  if (mr.size == 2) {
    lineToMsM(trr, mr.pts[kHead], mr.pts[kTail]);
    return;
  }
  if (mr.size == 4) {
    Trr trr_left;
    if (lineTypeM(mr.pts[0], mr.pts[1]) == LineType::kManhattan) {
      lineToMsM(trr_left, mr.pts[0], mr.pts[1]);
    } else {
      // LOG_FATAL_IF(lineType(mr[2], mr[1]) != LineType::kManhattan) << "mr is not manhattan";
      lineToMsM(trr_left, mr.pts[1], mr.pts[2]);
    }
    Trr trr_right;
    if (lineTypeM(mr.pts[2], mr.pts[3]) == LineType::kManhattan) {
      lineToMsM(trr_right, mr.pts[2], mr.pts[3]);
    } else {
      // LOG_FATAL_IF(lineType(mr[0], mr[3]) != LineType::kManhattan) << "mr is not manhattan";
      lineToMsM(trr_right, mr.pts[3], mr.pts[0]);
    }
    trr = trr_left;
    enclose(trr,trr_right);
    return;
  }
  // LOG_FATAL << "mr size is not 1, 2 or 4";
}
__device__ double calcDelayIncreaseM(const double& x, const double& y, const double& cap, const RCPattern& pattern){
  double delay = 0;
  switch (pattern) {
    case RCPattern::kHV:
      delay = _unit_h_resM * x * (_unit_h_capM * x / 2 + cap) + _unit_v_resM * y * (_unit_v_capM * y / 2 + cap + x * _unit_h_capM);
      break;
    case RCPattern::kVH:
      delay = _unit_v_resM * y * (_unit_v_capM * y / 2 + cap) + _unit_h_resM * x * (_unit_h_capM * x / 2 + cap + y * _unit_v_capM);
      break;
    case RCPattern::kSingle:
      delay = _unit_h_resM * (x + y) * (_unit_h_capM * (x + y) / 2 + cap);
      break;
    default:
      // LOG_FATAL << "unknown pattern";
      break;
  }
  return delay;
}
__device__ double ptDelayIncreaseM(Pt& p1, Pt& p2, const double& len, const double& cap, const RCPattern& pattern){
  auto h = fabs(p1.x - p2.x);
  auto v = fabs(p1.y - p2.y);
  // LOG_FATAL_IF(!Equal(len, h + v) && len < h + v) << "len is less than h + v";
  double delay = 0;
  if (EqualM(h, 0)) {
    delay = calcDelayIncreaseM(0, len, cap, pattern);
  } else if (EqualM(v, 0)) {
    delay = calcDelayIncreaseM(len, 0, cap, pattern);
  } else {
    delay = calcDelayIncreaseM(h, v, cap, pattern);
    if (len > h + v) {
      delay += calcDelayIncreaseM(0, len - h - v, cap + _unit_h_capM * h + _unit_v_capM * v, pattern);
    }
  }
  // LOG_FATAL_IF(delay < 0) << "point increase delay is negative";
  return delay;
}
__device__ void coreMidPointM(Trr& ms, Pt& mid){
  auto x = (ms._x_low + ms._x_high) / 2;
  auto y = (ms._y_low + ms._y_high) / 2;
  mid.x = (x + y) / 2;
  mid.y = (y - x) / 2;
}
__device__ double ptToLineDistManhattanM(Pt& p, const Line& l, Pt& closest){
  // manhattan arc
  Trr ms_p = Trr(p, 0);
  Trr ms_l;
  lineToMsM(ms_l, l);
  auto dist = msDistanceM(ms_p, ms_l);
  Trr new_ms_p;
  new_ms_p.makeDiamond(p, dist);
  Trr intersect_ms;
  makeIntersectM(new_ms_p, ms_l, intersect_ms);
  coreMidPointM(intersect_ms, closest);
  return dist;
}
__device__ double ptToLineDistNotManhattan(Pt& p, const Line& l, Pt& closest){
  size_t kHead = 0;
  size_t kTail = 1;
  Pt candidate[4];
  int cnt=0;
  if (!EqualM(l.pts[kHead].x, l.pts[kTail].x) && (p.x - l.pts[kHead].x) * (p.x - l.pts[kTail].x) <= 0) {
    Pt pt;
    pt.x = p.x;
    pt.y = (l.pts[kTail].x - p.x) * (l.pts[kHead].y - l.pts[kTail].y) / (l.pts[kTail].x - l.pts[kHead].x) + l.pts[kTail].y;
    candidate[cnt++]=pt;
  }
  if (!EqualM(l.pts[kHead].y, l.pts[kTail].y) && (p.y - l.pts[kHead].y) * (p.y - l.pts[kTail].y) <= 0) {
    Pt pt;
    pt.x = (l.pts[kTail].y - p.y) * (l.pts[kHead].x - l.pts[kTail].x) / (l.pts[kTail].y - l.pts[kHead].y) + l.pts[kTail].x;
    pt.y = p.y;
    candidate[cnt++]=pt;
  }
  if (cnt < 2) {
    candidate[cnt++]=l.pts[kHead];
    candidate[cnt++]=l.pts[kTail];
  }
  double min_dist = DOUBLE_MAXM;
  // std::ranges::for_each(candidate, [&p, &min_dist, &closest](auto& pt) {
  for(int i=0;i<cnt;i++){
    auto dist = distanceM(p, candidate[i]);
    if (dist < min_dist) {
      min_dist = dist;
      closest = candidate[i];
    }
  }
  return min_dist;
}
__device__ double ptToLineDistNotManhattanM(Pt& p, const Line& l, Pt& closest){
  Pt candidate[4];
  int cnt=0;
    size_t kHead = 0;
  size_t kTail = 1;
  if (!EqualM(l.pts[kHead].x, l.pts[kTail].x) && (p.x - l.pts[kHead].x) * (p.x - l.pts[kTail].x) <= 0) {
    Pt pt;
    pt.x = p.x;
    pt.y = (l.pts[kTail].x - p.x) * (l.pts[kHead].y - l.pts[kTail].y) / (l.pts[kTail].x - l.pts[kHead].x) + l.pts[kTail].y;
    candidate[cnt++]=pt;
  }
  if (!EqualM(l.pts[kHead].y, l.pts[kTail].y) && (p.y - l.pts[kHead].y) * (p.y - l.pts[kTail].y) <= 0) {
    Pt pt;
    pt.x = (l.pts[kTail].y - p.y) * (l.pts[kHead].x - l.pts[kTail].x) / (l.pts[kTail].y - l.pts[kHead].y) + l.pts[kTail].x;
    pt.y = p.y;
    candidate[cnt++]=pt;
  }
  if (cnt < 2) {
    candidate[cnt++]=l.pts[kHead];
    candidate[cnt++]=l.pts[kTail];
  }
  double min_dist = DOUBLE_MAXM;
  // std::ranges::for_each(candidate, [&p, &min_dist, &closest](auto& pt) {
  for(int i=0;i<cnt;i++){
    auto dist = distanceM(p, candidate[i]);
    if (dist < min_dist) {
      min_dist = dist;
      closest = candidate[i];
    }
  }
  return min_dist;
}
__device__ double ptToLineDistM(Pt& p, const Line& l, Pt& closest){
  auto min_dist = DOUBLE_MAXM;
  size_t kHead = 0;
  size_t kTail = 1;
  auto delta_x = fabs(l.pts[kHead].x - l.pts[kTail].x);
  auto delta_y = fabs(l.pts[kHead].y - l.pts[kTail].y);
  if (isSameM(l.pts[kHead], l.pts[kTail])) {
    closest = l.pts[kHead];
    min_dist = distanceM(p, closest);
  } else if (onLineM(p, l)) {
    closest = p;
    min_dist = 0;
  } else if (EqualM(delta_x, delta_y)) {
    // manhattan arc
    min_dist = ptToLineDistManhattanM(p, l, closest);
  } else {
    // not manhattan arc
    min_dist = ptToLineDistNotManhattanM(p, l, closest);
  }
  // LOG_FATAL_IF(onLine(closest, l) == false) << "closest point is not on line";
  return min_dist;
}
__global__ void embeddingChild(Area* areas, int side,int beginID,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) return;
  // printf("%d\n",id);
  int lrsize=areas[id+beginID].r-areas[id+beginID].l+1;
  // printf("%d\n",lrsize);
  if(lrsize<=1)return ;
  Pt child_loc;
  size_t kHead = 0;
  size_t kTail = 1;
  int parent=id+beginID;
  int child;
  if(side==0)child=areas[parent].left;
  else child=areas[parent].right;
  Pt parent_loc=areas[parent].location;
  // printf("%f %f\n",parent_loc.x,parent_loc.y);
  // auto mr = child->get_mr();

  if (areas[child].mr.size == 4 && isTrrArea(&areas[child])) {
    Trr trr;
    mrToTrr(areas[child].mr, trr);
    auto dist = ptToTrrDist(parent_loc, trr,DOUBLE_MAXM);
    Trr parent_trr(parent_loc, dist);
    Trr ms;
    makeIntersectM(parent_trr, trr, ms);
    coreMidPointM(ms, child_loc);
  } else {
    auto js_line = areas[parent].lines[side];
    auto head = js_line.pts[kHead];
    auto tail = js_line.pts[kTail];
    Line temp;
    calcBsLocatedM(&areas[child], head, temp);
    calcBsLocatedM(&areas[child], tail, temp);
    auto x = fabs(head.x - tail.x);
    auto y = fabs(head.y - tail.y);
    if (EqualM(x, 0) && EqualM(y, 0)) {
      // kHead loc is same as kTail loc
      child_loc = head;
    } else if (EqualM(x, 0)) {
      // vertical
      child_loc.x = head.x;
      child_loc.y = parent_loc.y;
    } else if (EqualM(y, 0)) {
      // horizontal
      child_loc.x = parent_loc.x;
      child_loc.y = head.y;
    } else {
      // others
      ptToLineDistM(parent_loc, js_line, child_loc);
    }
    // LOG_FATAL_IF(!Geom::onLine(child_loc, js_line)) << "child loc is not on js line";
  }
  areas[child].location=child_loc;
  if (areas[id+beginID]._edge_len[side] >= 0) {
    // LOG_FATAL_IF(parent->get_edge_len(side) < distanceM(parent_loc, child_loc) - kEpsilon) << "edge len is less than distance";
  } else {
    areas[id+beginID]._edge_len[side]=distanceM(parent_loc, child_loc);
    // parent->set_edge_len(side, distanceM(parent_loc, child_loc));
  }
}
__device__ double ptSkewM(const Pt& pt){
  return pt.max - pt.min;
}
__global__ void updateTiming(Area* areas,int beginID,int areasize){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=areasize)return;
  int size=areas[beginID+id].r-areas[beginID+id].l+1;
  if(size<=1)return;
  int kLeft=0,kRight=1;
  // update timing
  auto pt = areas[beginID+id].location;
  int left_index=areas[beginID+id].left;
  int right_index=areas[beginID+id].right;
  // auto left_pt = left->get_location();
  auto left_pt=areas[left_index].location;
  // auto right_pt = right->get_location();
  auto right_pt=areas[right_index].location;
  pt.min = DOUBLE_MAXM;
  pt.max = DOUBLE_MINM;
  auto delay_left = ptDelayIncreaseM(pt, left_pt, areas[beginID+id]._edge_len[kLeft], areas[left_index].cap, _patternM);
  auto delay_right = ptDelayIncreaseM(pt, right_pt, areas[beginID+id]._edge_len[kRight], areas[right_index].cap, _patternM);
  // printf("delay_left:%f,delay_right:%f\nright_pt:%f %f\nleft_pt:%f %f\ndelay:%f %f %f %f %f %f\n",delay_left,delay_right,right_pt.x,right_pt.y,left_pt.x,left_pt.y,pt.x,pt.y,areas[left_index].cap,areas[right_index].cap,areas[beginID+id]._edge_len[kLeft],areas[beginID+id]._edge_len[kRight]);
  pt.min = fmin(left_pt.min + delay_left, right_pt.min + delay_right);
  pt.max = fmax(left_pt.max + delay_left, right_pt.max + delay_right);
  // LOG_FATAL_IF(ptSkew(pt) > _skew_bound + 100 * kEpsilon) << "skew is so larger than skew bound, skew: " << ptSkew(pt);
  if (ptSkewM(pt) > _skew_boundM + kEpsilonM) {
    // LOG_WARNING << cur->get_name() << " max delay: " << pt.max << " min delay: " << pt.min;
    // LOG_WARNING << "skew is larger than skew bound with error: " << ptSkew(pt) - _skew_bound;
    pt.min = pt.max - _skew_boundM + kEpsilonM;
  }
  // cur->set_location(pt);
  areas[beginID+id].location=pt;
  // printf("min-max:%f %f\n",pt.min,pt.max);
}
__global__ void initroot(Area*areas,int beginID,int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id>=size)return;
  double X=0,Y=0;
  // printf("root:%f %f\n",areas[beginID].location.x,areas[beginID].location.y);
  for(int i=0;i<areas[beginID+id].mr.size;i++){
    X+=areas[beginID+id].mr.pts[i].x;
    Y+=areas[beginID+id].mr.pts[i].y;
  }
  areas[beginID+id].location=Pt(X/areas[beginID+id].mr.size,Y/areas[beginID+id].mr.size);
  areas[beginID+id].x=X/areas[beginID+id].mr.size;
  areas[beginID+id].y=Y/areas[beginID+id].mr.size;
  if(areas[beginID+id].mr.size==1)areas[beginID+id].x+=1.0,areas[beginID+id].location.x+=1.0;
  if(areas[beginID+id].mr.size>1){
    for(int i=areas[beginID+id].l;i<=areas[beginID+id].r;i++){
      if(distanceM(areas[i].location,areas[beginID+id].location)<1e-3){
        areas[beginID+id].x+=1.0,areas[beginID+id].location.x+=1.0;
        break;
      }
    }
    // printf("root:%d %f %f\n",areas[beginID+id].mr.size,areas[beginID+id].x,areas[beginID+id].y);
  }
  // printf("root:%f %f\n",areas[beginID+id].location.x,areas[beginID+id].location.y);
}
void BoundSkewTree::embedding(){
  double unit_h_cap=8.57718e-05;
  double unit_h_res=7.83333e-05;
  double unit_v_cap=7.595e-05;
  double unit_v_res=7.83333e-05;
  double skew_bound=0.;
  double kepsilon=1e-7;
  RCPattern pattern = RCPattern::kHV;
  Side<double> K(0.5 * unit_h_res * unit_h_cap, 0.5 * unit_v_res* unit_v_cap);
  double _DOUBLE_MAX=std::numeric_limits<double>::max();
  double _DOUBLE_MIN=std::numeric_limits<double>::min();
  cudaMemcpyToSymbol(_patternM, &pattern, sizeof(RCPattern));
  cudaMemcpyToSymbol(DOUBLE_MAXM, &_DOUBLE_MAX, sizeof(double));
  cudaMemcpyToSymbol(DOUBLE_MINM, &_DOUBLE_MIN, sizeof(double));
  cudaMemcpyToSymbol(_unit_h_capM, &unit_h_cap, sizeof(double));
  cudaMemcpyToSymbol(_unit_h_resM, &unit_h_res, sizeof(double));
  cudaMemcpyToSymbol(_unit_v_capM, &unit_v_cap, sizeof(double));
  cudaMemcpyToSymbol(_unit_v_resM, &unit_v_res, sizeof(double));
  cudaMemcpyToSymbol(_skew_boundM, &skew_bound, sizeof(double));
  cudaMemcpyToSymbol(kEpsilonM, &kepsilon, sizeof(double));

  int blockSize = 256;
  int size=net_num;
  int gridSize = (size + blockSize - 1) / blockSize;
  initroot<<<gridSize, blockSize>>>(areas,pinsize,size);
  for(int i=0;i<maxlevel;i++){
    int size=levelsum[i+1]-levelsum[i];
    gridSize = (size + blockSize - 1) / blockSize;
    // std::cout<<levelsum[i]<<' '<<size<<'\n';
    embeddingChild<<<gridSize, blockSize>>>(areas,0,levelsum[i],size);
    cudaDeviceSynchronize();
    embeddingChild<<<gridSize, blockSize>>>(areas,1,levelsum[i],size);
    cudaDeviceSynchronize();
  }
  for(int i=maxlevel;i>0;i--){
    int size=levelsum[i]-levelsum[i-1];
    gridSize = (size + blockSize - 1) / blockSize;
    // std::cout<<levelsum[i-1]<<' '<<size<<'\n';
    updateTiming<<<gridSize, blockSize>>>(areas,levelsum[i-1],size);
    cudaDeviceSynchronize();
  }
}
// Pt BoundSkewTreecpu::closestPtOnRegion(const Pt& p, const Region& region)
// {
//   if (isRegionContain(p, region)) {
//     return p;
//   }
//   auto pt = p;
//   Pt closest;
//   Pt ans;
//   auto min_dist = std::numeric_limits<double>::max();
//   for (size_t i = 0; i < region.size(); ++i) {
//     auto j = (i + 1) % region.size();
//     auto dist = ptToLineDist(pt, {region[i], region[j]}, closest);
//     if (dist < min_dist) {
//       min_dist = dist;
//       ans = closest;
//     }
//   }
//   return ans;
// }
}