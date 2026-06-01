// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file Pin.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
 #pragma once

 #include <iostream>
 #include <vector>
 #include <optional>
 #include <fstream> 
 #include <cuda_runtime.h>
 #include "Enumgpu.hh"
 namespace gcts {
 class Point {
 public:
     Point(){}
     Point(double _x, double _y) : __x(_x), __y(_y) {}
     double x(){return __x;}
     double y(){return __y;}
     double x() const { return __x; }  
     double y() const { return __y; }  
     bool operator==(const Point& rhs) const { return __x == rhs.__x && __y == rhs.__y; }
     bool operator!=(const Point& that) const { return !(*this == that); }
     static double manhattanDistance(const Point& p1, const Point& p2) { return std::abs(p1.x() - p2.x()) + std::abs(p1.y() - p2.y()); }
//  private:
     double __x, __y;
 };
class Pointc {
 public:
    __host__ __device__ Pointc(double _x, double _y,int _count) : x(_x), y(_y),count(_count) {}
    __host__ __device__ Pointc(double _x, double _y) : x(_x), y(_y) {count=1;cap=0;instID=-1;}
    __host__ __device__ Pointc(){x=0,y=0,count=0;cap=0;instID=-1;}
    __host__ __device__ Pointc(double _x, double _y,int _count,double _cap) : x(_x), y(_y),count(_count),cap(_cap) {}
    __host__ __device__ Pointc operator-(const Pointc& that) const { return Pointc(this->x - that.x, this->y - that.y,this->count,this->cap); }
    __host__ __device__ Pointc operator+(const Pointc& that) const { return Pointc(this->x + that.x, this->y + that.y,this->count,this->cap); }
    __host__ __device__ Pointc operator/(const int& that) const { return Pointc(this->x/that, this->y/that,this->count,this->cap); }
    __host__ __device__ Pointc operator*(const double& that) const { return Pointc(this->x*that, this->y*that,this->count,this->cap); }
     double x, y,cap;
     int count,clusterid,instID;
     __host__ __device__ bool operator<(const Pointc& other) const {
        return clusterid < other.clusterid; // 按key排序
    }
 };
 class Pin {
 public:
     Pin(const std::string& name, const Point& location) 
         : Pinname(name), location(location) {}  // 修正这里
     Pin(const Point& location, const std::string& pin_name, const PinType& pin_type = PinType::kLoad):location(location),pin_type(pin_type){}
    Point get_location(){return location;}
    void set_location(const Point& p){location=p;}
    std::string get_name(){return Pinname;}
    double get_cap_load(){return cap;}
    void set_cap_load(const double _cap){cap=_cap;}
 private:
     std::string Pinname;
     Point location;
     PinType pin_type;
     void init();
     double cap;
 };
 }  // namespace gcts