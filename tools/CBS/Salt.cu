#include "salt/salt_gpu.cuh"
#include "TreeMaker.h"
#include "Pingpu.hh"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <vector>

void _flute_cuda(uintptr_t num_nets, uint32_t *net_acc, const bool *skip_net,
                 const uintptr_t *net_start, const salt::cuda::DTYPE *x,
                 const salt::cuda::DTYPE *y, salt::cuda::DTYPE *flute_x,
                 salt::cuda::DTYPE *flute_y, uint32_t *flute_n,
                 uint32_t *flute_x_orig_id, uint32_t *flute_y_orig_id,
                 const char *lut_path);

void _salt_refine_cuda(uintptr_t num_nets, uintptr_t *net_start,
                       salt::cuda::DTYPE *net_x, salt::cuda::DTYPE *net_y,
                       uintptr_t *tree_start,uintptr_t *roots, salt::cuda::DTYPE *flute_x,
                       salt::cuda::DTYPE *flute_y, uint32_t *flute_n,
                       double epsilon, uintptr_t *salt_tree_start,
                       salt::cuda::DTYPE *salt_x, salt::cuda::DTYPE *salt_y,
                       uint32_t *salt_n);
void _bfs_cuda(uintptr_t num_nets, uintptr_t *net_start,
               salt::cuda::DTYPE *net_x, salt::cuda::DTYPE *net_y,
               uintptr_t *tree_start, salt::cuda::DTYPE *flute_x,
               salt::cuda::DTYPE *flute_y, uint32_t *flute_n, double epsilon,
               uint32_t *result_n, bool *breakpoint);
void _rsma_generation(intptr_t num_nets, uintptr_t *net_start,
                      salt::cuda::DTYPE *net_x, salt::cuda::DTYPE *net_y,
                      uintptr_t *tree_start, salt::cuda::DTYPE *tree_x,
                      salt::cuda::DTYPE *tree_y, uint32_t *tree_n,
                      bool *rsma_nodes, uintptr_t *result_tree_start,
                      salt::cuda::DTYPE *result_x, salt::cuda::DTYPE *result_y,
                      uint32_t *result_n);
struct Data {
    uint32_t d, acc;
    std::vector<salt::cuda::DTYPE> x;
    std::vector<salt::cuda::DTYPE> y;
};

__global__ void GetRootIdx(uintptr_t num_nets,
                           uintptr_t *__restrict__ net_start,
                           salt::cuda::DTYPE *__restrict__ net_x,
                           salt::cuda::DTYPE *__restrict__ net_y,
                           uintptr_t *__restrict__ tree_start,
                           salt::cuda::DTYPE *__restrict__ tree_x,
                           salt::cuda::DTYPE *__restrict__ tree_y,
                           uintptr_t *__restrict__ roots);

constexpr unsigned int BLOCK = 256;
inline unsigned int BLOCKS(int x) { return (x + BLOCK - 1) / BLOCK; }

void gcts::TreeMaker::saltgpu(int num_net,std::vector<int>segments, std::vector<int>tree_starts,std::vector<Point>pins,std::vector<Point>nodes,std::vector<int>_n,std::vector<int>roots,std::vector<bool>pin_ids)
{

    auto start = std::chrono::steady_clock::now();
    uintptr_t num_nets = num_net;
    std::string path = "/root/autodl-tmp/CTS/test.txt";
    auto fp = fopen(path.c_str(), "r");
    uint32_t *net_acc = nullptr;
    bool *skip_net = nullptr;
    uintptr_t *net_start = nullptr;
    salt::cuda::DTYPE *x = nullptr;
    salt::cuda::DTYPE *y = nullptr;

    net_acc = new uint32_t[num_nets];
    skip_net = new bool[num_nets];
    net_start = new uintptr_t[num_nets + 1];
    // net_start[0] = 0;
    // net_start[num_nets]=4;
    for(int i=0;i<segments.size();i++)net_start[i]=segments[i];
    x = new salt::cuda::DTYPE[net_start[num_nets]];
    y = new salt::cuda::DTYPE[net_start[num_nets]];

    int cnt=0;
    int xx,yy;
    double c;
    // while (fscanf(fp, "%d %d %f", &xx,&yy,&c) != EOF){
    //     x[cnt]=xx;
    //     y[cnt]=yy;
    //     cnt++;
    // }
    for(int i=0;i<pins.size();i++){
        x[i]=pins[i].x(), y[i]=pins[i].y();
    }
    salt::cuda::DTYPE *flute_x = new salt::cuda::DTYPE[net_start[num_nets] * 2];
    salt::cuda::DTYPE *flute_y = new salt::cuda::DTYPE[net_start[num_nets] * 2];
    uint32_t *flute_n = new uint32_t[net_start[num_nets] * 2];
    uintptr_t *flute_len = new uintptr_t[num_nets];
    uint32_t *flute_x_orig_id = new uint32_t[net_start[num_nets] * 2];
    uint32_t *flute_y_orig_id = new uint32_t[net_start[num_nets] * 2];
    std::cout<<"node_num "<<nodes.size()<<'\n';
    printf("num_nets %lu\n", num_nets);
    for(int i=0;i<nodes.size();i++){
        flute_x[i]=nodes[i].x(), flute_y[i]=nodes[i].y(),flute_n[i]=_n[i];
        // std::cout<<flute_x[i]<<' '<<flute_y[i]<<' '<<flute_n[i]<<'\n';
    }
    // bool *breakpoint= new bool[nodes.size()];
    // for(int i=0;i<nodes.size();i++)breakpoint[i]=pin_ids[i];
    // std::string pathr = "/root/autodl-tmp/CTS/testbst.txt";
    // auto fpr = fopen(pathr.c_str(), "r");
    // int X,Y,n;
    // int cc=0;
    // while (fscanf(fpr, "%d %d %d", &X,&Y,&n) != EOF){
    //     flute_x[cc]=X;
    //     flute_y[cc]=Y;
    //     flute_n[cc]=n;
    //     cc++;
    // }
    // read net above
    double epsilon = 0.2;

    uintptr_t *tree_start = nullptr;
    tree_start = new uintptr_t[num_nets + 1];
    for (uintptr_t i = 0; i < num_nets + 1; i++) {
        tree_start[i] = tree_starts[i];
    }

    uintptr_t *salt_tree_start = new uintptr_t[num_nets + 1];
    salt::cuda::DTYPE *salt_x = new salt::cuda::DTYPE[tree_start[num_nets] * 3];
    salt::cuda::DTYPE *salt_y = new salt::cuda::DTYPE[tree_start[num_nets] * 3];
    uint32_t *salt_n = new uint32_t[tree_start[num_nets] * 3];
    uintptr_t *root = new uintptr_t[num_nets + 1];
    // root[0]=0,root[1]=0;
    for(int i=0;i<roots.size();i++){
        root[i]=roots[i];
    }
    root[num_nets]=0;
    std::cout<<net_start[0]<<' '<<net_start[1]<<'\n';
    //{ _salt_refine_cuda(num_nets, net_start, x, y, tree_start, root, flute_x, flute_y,
    //                   flute_n, epsilon, salt_tree_start, salt_x, salt_y,
    //                   salt_n);
    //     for (uintptr_t i = 0; i < 1; i++) {
    //     uintptr_t l = salt_tree_start[i], size = salt_tree_start[i + 1] - l;
    //     // size = size > 20 ? 20 : size;
    //     printf("net %lu size %lu\n", i, salt_tree_start[i + 1] - l);
    //     for (uintptr_t j = salt_tree_start[i]; j < salt_tree_start[i + 1] - l; j++) {
    //         printf("%lu : %llu,%llu %lu\n", j, salt_x[l + j], salt_y[l + j],
    //                salt_n[l + j]);
    //     }
    //     printf("\n");
    // }
    // }

    uint32_t *bfs_result_n = new uint32_t[tree_start[num_nets]];
    // uint32_t *bfs_result_n=flute_n;
    bool *breakpoint = new bool[tree_start[num_nets]];
    _bfs_cuda(num_nets, net_start, x, y, tree_start, flute_x, flute_y, flute_n,
              epsilon, bfs_result_n, breakpoint);
    // std::cout<<flute_x[0]<<' '<<flute_y[0]<<'\n';
    // for(int i=tree_start[0];i<tree_start[1];i++)std::cout<<  breakpoint[i]<<' ';  
    // for(int i=tree_start[0]+1;i<tree_start[1];i++)breakpoint[i]=1;
        uintptr_t *result_tree_start = new uintptr_t[num_nets + 1];
    salt::cuda::DTYPE *result_x =
        new salt::cuda::DTYPE[tree_start[num_nets] * 3];
    salt::cuda::DTYPE *result_y =
        new salt::cuda::DTYPE[tree_start[num_nets] * 3];
    uint32_t *result_n = new uint32_t[tree_start[num_nets] * 3];

    _rsma_generation(num_nets, net_start, x, y, tree_start, flute_x, flute_y,
                     bfs_result_n, breakpoint, result_tree_start, result_x,
                     result_y, result_n);
    int c2=0;
    for (uintptr_t i = 0; i < 1; i++) {
        uintptr_t l = result_tree_start[i], size = result_tree_start[i + 1] - l;
        size = size > 20 ? 20 : size;
        printf("net %lu\n", i);
        for (uintptr_t j = 0; j < size; j++) {
            if(result_x[l + j]!=0)c2++;
            printf("%lu : %llu,%llu %u\n", j, result_x[l + j], result_y[l + j],
                   result_n[l + j]);
        }
        printf("\n");
    }
    std::cout<<c2<<'\n';
}

template <class T> void alloc_gpu(T *&ptr, size_t N)
{
    assert(ptr == nullptr);
    cudaMalloc(&ptr, sizeof(T) * N);
}

template <class T> void free_cpu(T *&ptr)
{
    if (ptr == nullptr)
        return;
    free(ptr);
    ptr = nullptr;
}

template <class T> void cpu2gpu(T *dst, T *src, size_t N)
{
    assert(dst != nullptr);
    assert(src != nullptr);
    cudaMemcpy(dst, src, sizeof(T) * N, cudaMemcpyHostToDevice);
}

template <class T> void cpu2gpu(T *dst, const T *src, size_t N)
{
    assert(dst != nullptr);
    assert(src != nullptr);
    cudaMemcpy(dst, src, sizeof(T) * N, cudaMemcpyHostToDevice);
}

template <class T> void gpu2cpu(T *dst, T *src, size_t N)
{
    assert(dst != nullptr);
    assert(src != nullptr);
    cudaMemcpy(dst, src, sizeof(T) * N, cudaMemcpyDeviceToHost);
}

template <class T> T get_gpu(T *ptr, size_t pos)
{
    assert(ptr != nullptr);
    T ret;
    cudaMemcpy(&ret, ptr + pos, sizeof(T), cudaMemcpyDeviceToHost);
    return ret;
}

void _flute_cuda(uintptr_t num_nets, uint32_t *net_acc, const bool *skip_net,
                 const uintptr_t *net_start, const salt::cuda::DTYPE *x,
                 const salt::cuda::DTYPE *y, salt::cuda::DTYPE *flute_x,
                 salt::cuda::DTYPE *flute_y, uint32_t *flute_n,
                 uint32_t *flute_x_orig_id, uint32_t *flute_y_orig_id,
                 const char *lut_path)
{
    uint32_t *d_net_acc = nullptr;
    bool *d_skip_net = nullptr;
    uintptr_t *d_net_start = nullptr;
    salt::cuda::DTYPE *d_x = nullptr, *d_y = nullptr, *d_result_x = nullptr,
                      *d_result_y = nullptr;
    uint32_t *d_result_n = nullptr, *d_result_x_orig_id = nullptr,
             *d_result_y_orig_id = nullptr;
    alloc_gpu(d_net_acc, num_nets);
    alloc_gpu(d_skip_net, num_nets);
    alloc_gpu(d_net_start, num_nets + 1);
    alloc_gpu(d_x, net_start[num_nets]);
    alloc_gpu(d_y, net_start[num_nets]);
    alloc_gpu(d_result_x, net_start[num_nets] * 2);
    alloc_gpu(d_result_y, net_start[num_nets] * 2);
    alloc_gpu(d_result_n, net_start[num_nets] * 2);
    alloc_gpu(d_result_x_orig_id, net_start[num_nets] * 2);
    alloc_gpu(d_result_y_orig_id, net_start[num_nets] * 2);
    cpu2gpu(d_net_acc, net_acc, num_nets);
    cpu2gpu(d_skip_net, skip_net, num_nets);
    cpu2gpu(d_net_start, net_start, num_nets + 1);
    cpu2gpu(d_x, x, net_start[num_nets]);
    cpu2gpu(d_y, y, net_start[num_nets]);

    printf("--------- start flute ---------\n");
    auto start = std::chrono::steady_clock::now();
    salt::cuda::flute_cuda(num_nets, d_net_acc, d_skip_net, d_net_start, d_x,
                           d_y, d_result_x, d_result_y, d_result_n,
                           d_result_x_orig_id, d_result_y_orig_id, lut_path);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("gpu flute %.2f\n", duration.count());
    printf("--------- end flute ---------\n\n");

    gpu2cpu(flute_x, d_result_x, net_start[num_nets] * 2);
    gpu2cpu(flute_y, d_result_y, net_start[num_nets] * 2);
    gpu2cpu(flute_n, d_result_n, net_start[num_nets] * 2);
    gpu2cpu(flute_x_orig_id, d_result_x_orig_id, net_start[num_nets] * 2);
    gpu2cpu(flute_y_orig_id, d_result_y_orig_id, net_start[num_nets] * 2);
}
void _bfs_cuda(uintptr_t num_nets, uintptr_t *net_start,
               salt::cuda::DTYPE *net_x, salt::cuda::DTYPE *net_y,
               uintptr_t *tree_start, salt::cuda::DTYPE *flute_x,
               salt::cuda::DTYPE *flute_y, uint32_t *flute_n, double epsilon,
               uint32_t *result_n, bool *breakpoint)
{
    uintptr_t *d_net_start = nullptr;
    salt::cuda::DTYPE *d_net_x = nullptr;
    salt::cuda::DTYPE *d_net_y = nullptr;
    uintptr_t *d_tree_start = nullptr;
    salt::cuda::DTYPE *d_flute_x = nullptr;
    salt::cuda::DTYPE *d_flute_y = nullptr;
    uint32_t *d_flute_n = nullptr;
    uint32_t *d_result_n = nullptr;
    bool *d_breakpoint = nullptr;
    alloc_gpu(d_net_start, num_nets + 1);
    alloc_gpu(d_net_x, net_start[num_nets]);
    alloc_gpu(d_net_y, net_start[num_nets]);
    alloc_gpu(d_tree_start, num_nets + 1);
    alloc_gpu(d_flute_x, tree_start[num_nets]);
    alloc_gpu(d_flute_y, tree_start[num_nets]);
    alloc_gpu(d_flute_n, tree_start[num_nets]);
    cpu2gpu(d_net_start, net_start, num_nets + 1);
    cpu2gpu(d_net_x, net_x, net_start[num_nets]);
    cpu2gpu(d_net_y, net_y, net_start[num_nets]);
    cpu2gpu(d_tree_start, tree_start, num_nets + 1);
    cpu2gpu(d_flute_x, flute_x, tree_start[num_nets]);
    cpu2gpu(d_flute_y, flute_y, tree_start[num_nets]);
    cpu2gpu(d_flute_n, flute_n, tree_start[num_nets]);

    alloc_gpu(d_result_n, tree_start[num_nets]);
    alloc_gpu(d_breakpoint, tree_start[num_nets]);

    uintptr_t *roots = nullptr;
    alloc_gpu(roots, num_nets);
    GetRootIdx<<<BLOCKS(num_nets), BLOCK>>>(num_nets, d_net_start, d_net_x,
                                            d_net_y, d_tree_start, d_flute_x,
                                            d_flute_y, roots);
#ifndef NDEBUG
    cudaDeviceSynchronize();
#endif

    printf("--------- start bfs ---------\n");

    auto start = std::chrono::high_resolution_clock::now();
    salt::cuda::breakpoint_detection_cuda(
        num_nets, d_net_start, d_net_x, d_net_y, d_tree_start, roots, d_flute_x,
        d_flute_y, d_flute_n, epsilon, d_result_n, d_breakpoint);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time = end - start;
    printf("bfs %.2f\n", time.count());
    printf("--------- end bfs ---------\n\n");

    gpu2cpu(result_n, d_result_n, tree_start[num_nets]);
    gpu2cpu(breakpoint, d_breakpoint, tree_start[num_nets]);
}

void _salt_refine_cuda(uintptr_t num_nets, uintptr_t *net_start,
                       salt::cuda::DTYPE *net_x, salt::cuda::DTYPE *net_y,
                       uintptr_t *tree_start, uintptr_t *roots, salt::cuda::DTYPE *flute_x,
                       salt::cuda::DTYPE *flute_y, uint32_t *flute_n,
                       double epsilon, uintptr_t *salt_tree_start,
                       salt::cuda::DTYPE *salt_x, salt::cuda::DTYPE *salt_y,
                       uint32_t *salt_n)
{
    uintptr_t *d_net_start = nullptr;
    salt::cuda::DTYPE *d_net_x = nullptr;
    salt::cuda::DTYPE *d_net_y = nullptr;
    uintptr_t *d_tree_start = nullptr;
    uintptr_t *d_roots = nullptr;
    salt::cuda::DTYPE *d_flute_x = nullptr;
    salt::cuda::DTYPE *d_flute_y = nullptr;
    uint32_t *d_flute_n = nullptr;
    uintptr_t *d_salt_tree_start = nullptr;
    salt::cuda::DTYPE *d_salt_x = nullptr;
    salt::cuda::DTYPE *d_salt_y = nullptr;
    uint32_t *d_salt_n = nullptr;
    alloc_gpu(d_net_start, num_nets + 1);
    alloc_gpu(d_net_x, net_start[num_nets]);
    alloc_gpu(d_net_y, net_start[num_nets]);
    alloc_gpu(d_tree_start, num_nets + 1);
    alloc_gpu(d_roots, num_nets+1);
    alloc_gpu(d_flute_x, tree_start[num_nets]);
    alloc_gpu(d_flute_y, tree_start[num_nets]);
    alloc_gpu(d_flute_n, tree_start[num_nets]);
    cpu2gpu(d_net_start, net_start, num_nets + 1);
    cpu2gpu(d_net_x, net_x, net_start[num_nets]);
    cpu2gpu(d_net_y, net_y, net_start[num_nets]);
    cpu2gpu(d_tree_start, tree_start, num_nets + 1);
    cpu2gpu(d_roots, roots, num_nets + 1);
    cpu2gpu(d_flute_x, flute_x, tree_start[num_nets]);
    cpu2gpu(d_flute_y, flute_y, tree_start[num_nets]);
    cpu2gpu(d_flute_n, flute_n, tree_start[num_nets]);

    alloc_gpu(d_salt_tree_start, num_nets + 1);
    alloc_gpu(d_salt_x, tree_start[num_nets] * 3);
    alloc_gpu(d_salt_y, tree_start[num_nets] * 3);
    alloc_gpu(d_salt_n, tree_start[num_nets] * 3);

    // GetRootIdx<<<BLOCKS(num_nets), BLOCK>>>(num_nets, d_net_start, d_net_x,
    //                                         d_net_y, d_tree_start, d_flute_x,
    //                                         d_flute_y, roots);
#ifndef NDEBUG
    cudaDeviceSynchronize();
#endif

    printf("--------- start salt refine ---------\n");

    auto start = std::chrono::high_resolution_clock::now();
    salt::cuda::salt_refine_cuda(num_nets, d_net_start, d_net_x, d_net_y,
                                 d_tree_start, d_roots, d_flute_x, d_flute_y,
                                 d_flute_n, epsilon, d_salt_tree_start,
                                 d_salt_x, d_salt_y, d_salt_n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time = end - start;
    printf("refine salt flute %.2f\n", time.count());
    printf("--------- end salt refine ---------\n\n");

    gpu2cpu(salt_tree_start, d_salt_tree_start, num_nets + 1);
    gpu2cpu(salt_x, d_salt_x, tree_start[num_nets] * 3);
    gpu2cpu(salt_y, d_salt_y, tree_start[num_nets] * 3);
    gpu2cpu(salt_n, d_salt_n, tree_start[num_nets] * 3);
}

__global__ void GetRootIdx(uintptr_t num_nets,
                           uintptr_t *__restrict__ net_start,
                           salt::cuda::DTYPE *__restrict__ net_x,
                           salt::cuda::DTYPE *__restrict__ net_y,
                           uintptr_t *__restrict__ tree_start,
                           salt::cuda::DTYPE *__restrict__ tree_x,
                           salt::cuda::DTYPE *__restrict__ tree_y,
                           uintptr_t *__restrict__ roots)
{
    uintptr_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_nets)
        return;
    uintptr_t net_left = net_start[idx];
    net_x += net_left;
    net_y += net_left;
    uintptr_t tree_left = tree_start[idx],
              tree_size = tree_start[idx + 1] - tree_left;
    tree_x += tree_left;
    tree_y += tree_left;

    for (uintptr_t i = 0; i < tree_size; i++) {
        if (tree_x[i] == net_x[0] && tree_y[i] == net_y[0]) {
            roots[idx] = tree_left + i;
            return;
        }
    }
}
void _rsma_generation(intptr_t num_nets, uintptr_t *net_start,
                      salt::cuda::DTYPE *net_x, salt::cuda::DTYPE *net_y,
                      uintptr_t *tree_start, salt::cuda::DTYPE *tree_x,
                      salt::cuda::DTYPE *tree_y, uint32_t *tree_n,
                      bool *rsma_nodes, uintptr_t *result_tree_start,
                      salt::cuda::DTYPE *result_x, salt::cuda::DTYPE *result_y,
                      uint32_t *result_n)
{

    uintptr_t *d_tree_start = nullptr;
    salt::cuda::DTYPE *d_tree_x = nullptr;
    salt::cuda::DTYPE *d_tree_y = nullptr;
    uint32_t *d_tree_n = nullptr;
    bool *d_rsma_nodes = nullptr;
    uintptr_t *d_result_tree_start = nullptr;
    salt::cuda::DTYPE *d_result_x = nullptr;
    salt::cuda::DTYPE *d_result_y = nullptr;
    uint32_t *d_result_n = nullptr;
    alloc_gpu(d_tree_start, num_nets + 1);
    alloc_gpu(d_tree_x, tree_start[num_nets]);
    alloc_gpu(d_tree_y, tree_start[num_nets]);
    alloc_gpu(d_tree_n, tree_start[num_nets]);
    alloc_gpu(d_rsma_nodes, tree_start[num_nets]);
    alloc_gpu(d_result_tree_start, num_nets + 1);
    alloc_gpu(d_result_x, tree_start[num_nets] * 3);
    alloc_gpu(d_result_y, tree_start[num_nets] * 3);
    alloc_gpu(d_result_n, tree_start[num_nets] * 3);
    cpu2gpu(d_tree_start, tree_start, num_nets + 1);
    cpu2gpu(d_tree_x, tree_x, tree_start[num_nets]);
    cpu2gpu(d_tree_y, tree_y, tree_start[num_nets]);
    cpu2gpu(d_tree_n, tree_n, tree_start[num_nets]);
    cpu2gpu(d_rsma_nodes, rsma_nodes, tree_start[num_nets]);

    uintptr_t *d_net_start = nullptr;
    salt::cuda::DTYPE *d_net_x = nullptr;
    salt::cuda::DTYPE *d_net_y = nullptr;
    alloc_gpu(d_net_start, num_nets + 1);
    alloc_gpu(d_net_x, net_start[num_nets]);
    alloc_gpu(d_net_y, net_start[num_nets]);
    cpu2gpu(d_net_start, net_start, num_nets + 1);
    cpu2gpu(d_net_x, net_x, net_start[num_nets]);
    cpu2gpu(d_net_y, net_y, net_start[num_nets]);

    uintptr_t *d_roots = nullptr;
    alloc_gpu(d_roots, num_nets);
    GetRootIdx<<<BLOCKS(num_nets), BLOCK>>>(num_nets, d_net_start, d_net_x,
                                            d_net_y, d_tree_start, d_tree_x,
                                            d_tree_y, d_roots);
#ifndef NDEBUG
    cudaDeviceSynchronize();
#endif

    printf("--------- generate rsma ---------\n");
    auto start = std::chrono::high_resolution_clock::now();
    salt::cuda::rsma_generation_cuda(
        num_nets, d_tree_start, d_roots, d_tree_x, d_tree_y, d_tree_n,
        d_rsma_nodes, d_result_tree_start, d_result_x, d_result_y, d_result_n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time = end - start;
    printf("rsma %.2f\n", time.count());
    printf("--------- rsma generated ---------\n");

    gpu2cpu(result_tree_start, d_result_tree_start, num_nets + 1);
    gpu2cpu(result_x, d_result_x, tree_start[num_nets] * 3);
    gpu2cpu(result_y, d_result_y, tree_start[num_nets] * 3);
    gpu2cpu(result_n, d_result_n, tree_start[num_nets] * 3);
}

