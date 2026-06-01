#include <iostream>
#include "GPUClustering.h"
#include <chrono> 
#include "auction.h" // 假设头文件路径正确
#include <cmath>  
namespace gcts{
double x[5000],y[5000];
double bufferx[5000],buffery[5000];    
void read(Pointc* pts,Pointc* buffers,int k,int num_nodes){
    Pointc h_pts[4380],h_buffers[4380];
    cudaMemcpy(h_pts, pts, 
               4380 * sizeof(Pointc), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_buffers, buffers, 
                k* sizeof(Pointc), cudaMemcpyDeviceToHost);
    for(int i=0;i<4380;i++)x[i]=h_pts[i].x,y[i]=h_pts[i].y;
    for(int i=4380;i<num_nodes;i++)x[i]=(double)i/10.,y[i]=(double)i/10.;
    for(int i=0;i<k;i++)bufferx[i]=h_buffers[i].x,buffery[i]=h_buffers[i].y;
    // std::cout<<cnt<<'\n';
}
double dist(int i,int j){
    return 3000.-(std::fabs(x[i]-bufferx[j])+std::fabs(y[i]-buffery[j]));
}    
void GPUClustering::AuctionRun(Pointc* pts,Pointc* clusters,Pointc* buffers,int *clusters_segment,double * cost_matrices,int max_fanout,int size){
    // 参数设置
    int k=std::ceil(1.0 * size / max_fanout);
    const int num_graphs = 1;      
    const int num_nodes = k*max_fanout;      
    const double max_eps = 50;   // 初始ε
    const double min_eps = 1.;    // 最小ε
    const double factor = 0.96;     // ε衰减因子
    const int max_iter = 800;      // 最大迭代次数
    read(pts,buffers,k,num_nodes);
    std::cout<<"Auction-max_eps:"<<max_eps<<' '<<num_nodes<<'\n';
    // 1. 准备数据（示例：随机生成成本矩阵）
    // double* h_cost_matrices = new double[num_graphs * num_nodes * num_nodes];
    // // cudaMemcpy(h_cost_matrices, cost_matrices, 
    // //            num_graphs * num_nodes * num_nodes * sizeof(double), cudaMemcpyDeviceToHost);
    // // // std::ofstream outfile("/root/autodl-tmp/keans/data/aution_cost1.txt");
    // for (int i = 0; i < num_graphs; ++i) {
    //     for (int j = 0; j < num_nodes; ++j) {
    //         for (int k = 0; k < num_nodes; k+=32) {
    //             // 随机收益值（实际应用中替换为真实数据）
    //             h_cost_matrices[i * num_nodes * num_nodes + j * num_nodes + k] = dist(j,k/32);
    //             //     static_cast<double>(rand() % 100);
    //             // printf("node_%d,node_%d:%f\n",j,k,h_cost_matrices[i * num_nodes * num_nodes + j * num_nodes + k]);
    //             // outfile<<h_cost_matrices[i * num_nodes * num_nodes + j * num_nodes + k]<<' ';
    //             // if(j>4098&&j<4100&&k>4300)printf("node_%d,node_%d:%f\n",j,k,h_cost_matrices[i * num_nodes * num_nodes + j * num_nodes + k]);
    //             for(int k1=k+1;k1<k+32&&k1<num_nodes;k1++){
    //                 h_cost_matrices[i * num_nodes * num_nodes + j * num_nodes + k1]=h_cost_matrices[i * num_nodes * num_nodes + j * num_nodes + k]+static_cast<double>(rand() % 32);
    //                 // if(j<2100&&j>2000&&k1>4300)printf("%d node_%d,node_%d:%f %f\n",i * num_nodes * num_nodes + j * num_nodes + k1,j,k1,cos,h_cost_matrices[i * num_nodes * num_nodes + j * num_nodes + k1]);
    //             }
    //         }
    //         // outfile<<'\n';
    //     }
    // }

    // 2. 分配GPU内存
    double * d_cost_matrices;
    int* d_solutions;
    char *scratch, *stop_flags;

    // cudaMalloc(&d_cost_matrices, num_graphs * num_nodes * num_nodes * sizeof(double));
    cudaMalloc(&d_solutions, num_graphs * num_nodes * sizeof(int));
    init_auction(num_graphs, num_nodes, scratch, stop_flags);

    // 3. 拷贝数据到GPU
    // cudaMemcpy(h_cost_matrices, cost_matrices, 
    //            num_graphs * num_nodes * num_nodes * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(d_cost_matrices, h_cost_matrices, 
    //            num_graphs * num_nodes * num_nodes * sizeof(double), cudaMemcpyHostToDevice);
    // 4. 调用拍卖算法

    auto start = std::chrono::high_resolution_clock::now();
    linear_assignment_auction(
        cost_matrices, d_solutions, num_graphs, num_nodes,
        scratch, stop_flags, max_eps, min_eps, factor, max_iter
    );
        auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "auction calculate time: " << elapsed.count() << " seconds" << std::endl;
    // 5. 取回结果
    int* h_solutions = new int[num_graphs * num_nodes];
    cudaMemcpy(h_solutions, d_solutions, 
               num_graphs * num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    double cost=0;
    // for (int i = 0; i < num_graphs; ++i) {
    //     for (int j = 0; j < size; ++j) {
    //         // if(h_solutions[j]<0||h_solutions[j]>=num_nodes)std::cout<<"errer\n";
    //         if(j<10)std::cout << "Person " << j << " -> Item " << h_solutions[i * num_nodes + j] << "\n";
    //         // cost+=(h_max-h_cost_matrices[num_nodes*j+h_solutions[j]]);
    //     }
    // }
    // std::cout<<"pts\n";
    // coutPointc(pts,10);
    // std::cout<<"buffers\n";
    // coutPointc(buffers,k);
    // std::cout<<"h_cost_matrices\n";
    // int nodeid=2,bid=2;
    // for(int i=0;i<k;i++)std::cout<<h_max-h_cost_matrices[nodeid*num_nodes+i*max_fanout]<<'\n';
    // std::cout<<"Auction-cost:"<<cost<<'\n';
    AuctionToCluster(pts,clusters,buffers,clusters_segment,d_solutions,max_fanout,size);
    // 6. 释放资源
    destroy_auction(scratch, stop_flags);
    // cudaFree(d_cost_matrices);
    cudaFree(d_solutions);
    delete[] h_solutions;
}
}