#include "salt/salt_gpu.cuh"
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

int main(int argc, char **argv)
{

    auto start = std::chrono::steady_clock::now();
    uintptr_t num_nets = 0;
    std::vector<Data> dataset;
    // std::vector<std::string> netnames({"1.net", "3.net", "4.net", "5.net",
    //                                   "7.net", "10.net", "16.net",
    //                                   "18.net"});
    std::vector<std::string> netnames({".net"});
    std::string rootpath = "../benchmarks/summon";
    for (uintptr_t i = 0; i < netnames.size(); i++) {
        std::string path = rootpath + netnames[i];
        auto fp = fopen(path.c_str(), "r");
        int d, acc;
        while (fscanf(fp, "%d %d", &d, &acc) != EOF) {
            // while (fscanf(fp, "%d", &d) != EOF) {
            dataset.emplace_back();
            auto &data = dataset.back();
            data.d = d;
            data.acc = 3;
            data.x.resize(d);
            data.y.resize(d);
            int ret;
            for (int i = 0; i < d; i++)
                ret = fscanf(fp, "%llu", &data.x[i]);
            for (int i = 0; i < d; i++)
                ret = fscanf(fp, "%llu", &data.y[i]);
            num_nets++;
            if (ret == EOF)
                printf("read error\n");
        }
        fclose(fp);
    }

    uint32_t *net_acc = nullptr;
    bool *skip_net = nullptr;
    uintptr_t *net_start = nullptr;
    salt::cuda::DTYPE *x = nullptr;
    salt::cuda::DTYPE *y = nullptr;

    net_acc = new uint32_t[num_nets];
    skip_net = new bool[num_nets];
    net_start = new uintptr_t[num_nets + 1];
    net_start[0] = 0;
    for (int i = 0; i < num_nets; i++) {
        net_acc[i] = dataset[i].acc;
        if (net_acc[i] == 999)
            skip_net[i] = true;
        else
            skip_net[i] = false;
        net_start[i + 1] = dataset[i].d;
    }
    for (int i = 1; i <= num_nets; i++)
        net_start[i] += net_start[i - 1];
    x = new salt::cuda::DTYPE[net_start[num_nets]];
    y = new salt::cuda::DTYPE[net_start[num_nets]];
    for (int i = 0; i < num_nets; i++) {
        auto &data = dataset[i];
        for (int j = 0; j < data.d; j++) {
            x[net_start[i] + j] = data.x[j];
            y[net_start[i] + j] = data.y[j];
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "data prepare in: " << duration.count() << " ms" << std::endl;
    std::cout << std::endl;

    salt::cuda::DTYPE *flute_x = new salt::cuda::DTYPE[net_start[num_nets] * 2];
    salt::cuda::DTYPE *flute_y = new salt::cuda::DTYPE[net_start[num_nets] * 2];
    uint32_t *flute_n = new uint32_t[net_start[num_nets] * 2];
    uintptr_t *flute_len = new uintptr_t[num_nets];
    uint32_t *flute_x_orig_id = new uint32_t[net_start[num_nets] * 2];
    uint32_t *flute_y_orig_id = new uint32_t[net_start[num_nets] * 2];

    printf("num_nets %lu\n", num_nets);

    _flute_cuda(num_nets, net_acc, skip_net, net_start, x, y, flute_x, flute_y,
                flute_n, flute_x_orig_id, flute_y_orig_id, "./flute-gpu/");

    // read net above
    double epsilon = 0.2;
    if (argc > 1) {
        std::stringstream ss;
        ss << argv[1];
        ss >> epsilon;
        std::cout << "epsilon: " << epsilon << std::endl;
    }
    uintptr_t *tree_start = nullptr;
    tree_start = new uintptr_t[num_nets + 1];
    for (uintptr_t i = 0; i < num_nets + 1; i++) {
        tree_start[i] = net_start[i] * 2;
    }

    uint32_t *bfs_result_n = new uint32_t[tree_start[num_nets]];
    bool *breakpoint = new bool[tree_start[num_nets]];

    _bfs_cuda(num_nets, net_start, x, y, tree_start, flute_x, flute_y, flute_n,
              epsilon, bfs_result_n, breakpoint);

    uintptr_t *result_tree_start = new uintptr_t[num_nets + 1];
    salt::cuda::DTYPE *result_x =
        new salt::cuda::DTYPE[tree_start[num_nets] * 3];
    salt::cuda::DTYPE *result_y =
        new salt::cuda::DTYPE[tree_start[num_nets] * 3];
    uint32_t *result_n = new uint32_t[tree_start[num_nets] * 3];

    _rsma_generation(num_nets, net_start, x, y, tree_start, flute_x, flute_y,
                     bfs_result_n, breakpoint, result_tree_start, result_x,
                     result_y, result_n);

    for (uintptr_t i = 0; i < 2; i++) {
        uintptr_t l = result_tree_start[i], size = result_tree_start[i + 1] - l;
        size = size > 20 ? 20 : size;
        printf("net %lu\n", i);
        for (uintptr_t j = 0; j < size; j++) {
            printf("%lu : %llu,%llu %u\n", j, result_x[l + j], result_y[l + j],
                   result_n[l + j]);
        }
        printf("\n");
    }
    return 0;
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
