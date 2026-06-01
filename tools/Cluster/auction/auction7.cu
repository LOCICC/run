#include <cstdlib>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "auction.h" 

// #include "utility/src/utils.cuh"
#define BIG_NEGATIVE    -9999999
#define MAX_MINIBATCH         64
#define allocateCUDA(var, size, type){cudaError_t status = cudaMalloc(&(var), (size) * sizeof(type));}
#define destroyCUDA(var){cudaError_t status = cudaFree(var);}
namespace gcts{
void init_auction(
        const int num_graphs, 
        const int num_nodes, 
        char*& scratch, 
        char*& stop_flags
        )
{
    cudaMalloc(&scratch, 
                num_graphs*(3*num_nodes+1)*sizeof(int) + num_graphs*(num_nodes*num_nodes+num_nodes)*sizeof(double)
                );
    allocateCUDA(stop_flags, num_graphs, char);
}
void destroy_auction(
        char* scratch, 
        char* stop_flags
        )
{
    destroyCUDA(scratch);
    destroyCUDA(stop_flags); 
}

__global__ void compute_orig_cost_kernel(const double* cost_matrices, const int num_nodes, double* costs)
{
    int i = blockIdx.x; // set 
    auto cost_matrix = cost_matrices + i*num_nodes*num_nodes; 
    for (int j = threadIdx.x; j < num_nodes; j += blockDim.x)
    {
        atomicAdd(costs+i, cost_matrix[j*num_nodes+j]);
    }
}

template <typename T>
__global__ void compute_solution_cost_kernel(const T* cost_matrices, const int* solutions, const int num_nodes, T* costs)
{
    int i = blockIdx.x; // set 
    auto cost_matrix = cost_matrices + i*num_nodes*num_nodes; 
    auto solution = solutions + i*num_nodes; 
    for (int j = threadIdx.x; j < num_nodes; j += blockDim.x)
    {
        atomicAdd(costs+i, cost_matrix[j*num_nodes+solution[j]]);
    }
}

template <typename T>
__global__ void print_costs_kernel(T* a, int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        printf("[%d]\n", n);
        for (int i = 0; i < n; ++i)
        {
            printf("%g ", (double)a[i]);
        }
        printf("\n");
    }
}

template <typename T>
__global__ void check_costs_kernel(const T* a, const T* b, int n, T epsilon)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        for (int i = 0; i < n; ++i)
        {
            if (a[i] > b[i]+epsilon)
            {
                printf("cost error %g > %g\n", a[i], b[i]);
            }
            assert(a[i] <= b[i]+epsilon);
        }
    }
}

template <typename T>
__global__ void print_solution_kernel(const T* solutions, int num_graphs, int num_nodes)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
    if (tid == 0 && bid == 0)
    {
        for (int i = 0; i < num_graphs; ++i)
        {
            printf("[%d]\n", i);
            for (int j = 0; j < num_nodes; ++j)
            {
                printf("%d ", solutions[i*num_nodes+j]);
            }
            printf("\n");
        }
    }
}

__global__ void print_stop_flags_kernel(char* stop_flags, int n)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("[%dx64]\n", n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < 64; ++j)
            {
                printf("%d", stop_flags[i]);
            }
            printf("\n");
        }
    }
}
__device__ void detail(int i,int &top1_col,int &top2_col,double &top1_val,double &top2_val,const double*data,double*s_prices,int num_nodes){
    int top1_col1=top1_col;
    int top2_col1=top2_col;
    double tmp_val;
    for (int col = top1_col; col < num_nodes&&col<top1_col+32; col++)
    {
        tmp_val = data[i * num_nodes + col]; 
         if (tmp_val < 0){
            continue;
        }
        tmp_val = tmp_val - s_prices[col];
        if (tmp_val >= top1_val){
            top2_val = top1_val;
            top1_col1 = col;
            top1_val = tmp_val;
        }
        else if (tmp_val > top2_val){
            top2_val = tmp_val;
            top2_col1 = col;
        }
    }
    for (int col = top2_col; col < num_nodes&&col<top2_col+32; col++)
    {
        tmp_val = data[i * num_nodes + col]; 
         if (tmp_val < 0){
            continue;
        }
        tmp_val = tmp_val - s_prices[col];
        if (tmp_val >= top1_val){
            top2_val = top1_val;
            top1_col1 = col;
            top1_val = tmp_val;
        }
        else if (tmp_val > top2_val){
            top2_val = tmp_val;
            top2_col1 = col;
        }
    }
    top1_col=top1_col1,top2_col=top2_col1;
}

void __global__ resetkey(int* person2item1,int* person2item,int* key_person,int* key_item){
    int id = threadIdx.x;
    person2item[key_person[id]]=key_item[person2item1[id]];
}
template <typename T>
__global__ void 
linear_assignment_auction_kernel(const int num_nodes,
                                        const T* __restrict__ data_ptr,
                                        int* person2item_ptr, 
                                        int*  item2person_ptr,
                                        double *s_prices,
                                        T*  bids_ptr,
                                        T*  prices_ptr,
                                        // int* sbids_ptr,
                                        char* stop_flag_ptr,
                                        const double auction_max_eps,
                                        const double auction_min_eps,
                                        const double auction_factor,
                                        const int max_iterations)
{
    int node_id = threadIdx.x;
    __shared__ double auction_eps;
    __shared__ int num_iteration;
    __shared__ int num_assigned;
    __shared__ int sbids[5000];
    __shared__ int prices_group[200];
    // extern __shared__ T s_prices[];
    
    if(node_id == 0){
        auction_eps = auction_max_eps;
        num_iteration = 0;
        num_assigned=0;
    }

    const T* __restrict__ data = data_ptr;
    int* person2item = person2item_ptr; 
    int*  item2person = item2person_ptr;
    T* bids = bids_ptr;
    // int* sbids = sbids_ptr;
    T*  prices = prices_ptr;
    char* stop_flag = stop_flag_ptr;
    __syncthreads();
    while(auction_eps >= auction_min_eps && num_assigned < num_nodes&&num_iteration < max_iterations)
    {    
        //clear num_assigned
        if(node_id == 0){
            num_assigned = 0;
            for(int i=0;i<num_nodes/32;i++)prices_group[i]=0;
        }
        //pre-init
        for(int i = node_id; i < num_nodes; i += blockDim.x){
            person2item[i] = -1;
            item2person[i] = -1;
        }
        // if(node_id%32==0&&node_id<num_nodes/32)prices_group[node_id/32]=0;
        __syncthreads();
        //start iterative solving
        while(num_assigned < num_nodes && num_iteration < max_iterations)
        {
            //phase 1: init bid and bids
            for(int i = node_id; i < num_nodes; i += blockDim.x){
                sbids[i] = 0;
            }
            for(int i = node_id; i < num_nodes*num_nodes; i += blockDim.x){
                bids[i] = 0;
            }
            //preload price
            for(int i = node_id; i < num_nodes; i += blockDim.x){
                s_prices[i] = prices[i];
                atomicMax(prices_group + i/32, (int)s_prices[i]);
            }
            __syncthreads();
            // if(node_id==0){
            //     for(int i=0;i<num_nodes/32;i++)printf("%d ",prices_group[i]);
            // }
            //phase 2: bidding
            for(int i = node_id; i < num_nodes; i += blockDim.x){
            if(person2item[i] == -1){
                T top1_val = BIG_NEGATIVE; 
                T top2_val = BIG_NEGATIVE; 
                int top1_col,top2_col; 
                T tmp_val;
 
                for (int col = 0; col < num_nodes; col+=32)
                {
                    tmp_val = data[i * num_nodes + col]; 
                    if (tmp_val < 0)
                    {
                        continue;
                    }
                    // tmp_val = tmp_val - s_prices[col];
                    tmp_val = tmp_val - (double)prices_group[col/32];
                    if (tmp_val >= top1_val)
                    {
                        top2_val = top1_val;
                        top1_col = col;
                        top1_val = tmp_val;
                    }
                    else if (tmp_val > top2_val)
                    {
                        top2_val = tmp_val;
                        top2_col = col;
                    }
                }
                // if(i==0)printf("auction %d %d %f %f\n",top1_col,top2_col,top1_val,top2_val);
                detail(i,top1_col,top2_col,top1_val,top2_val,data,s_prices,num_nodes);

                if (top2_val == BIG_NEGATIVE)
                {
                    top2_val = top1_val;
                }
                T bid = top1_val - top2_val + auction_eps;
                bids[num_nodes * top1_col + i] = bid;
                atomicMax(sbids + top1_col, ((int)bid)*10000+i);
                // if(i==0)printf("auction7 %d\n",((int)bid)*10000+i);
            }
            }

            __syncthreads();

            //phase 3 : assignment
            for(int j = node_id; j < num_nodes; j += blockDim.x){
            if(sbids[j] != 0) {
                T high_bid  = 0;
                int high_bidder = -1;
                // T tmp_bid = -1;
                // for(int i = 0; i < num_nodes; i++){
                //     tmp_bid = bids[j * num_nodes + i];
                //     if(tmp_bid > high_bid){
                //         high_bid    = tmp_bid;
                //         high_bidder = i;
                //     }
                // }
                high_bid = sbids[j] /10000.;
                high_bidder = sbids[j] %10000;
                int current_person = item2person[j];
                if(current_person >= 0){
                    person2item[current_person] = -1;
                } else {
                    atomicAdd(&num_assigned, 1);
                }
    
                prices[j]                += high_bid;
                person2item[high_bidder] = j;
                item2person[j]           = high_bidder;
            }
            }
            // __syncthreads();
            
            //update iteration
            if(node_id == 0){
                num_iteration++;
                // printf("iter%d num_assigned:%d\n",num_iteration,num_assigned);
            }
            __syncthreads();
        }
        //scale auction_eps
        if(node_id == 0){
            auction_eps *= auction_factor;
        }
        __syncthreads();
    }
    __syncthreads();
    //report whether finish solving
    // if(node_id == 0){
    //     // printf("num_assigned:%d num_nodes:%d %d\n",num_assigned,num_nodes,num_iteration);
    //     *stop_flag = (num_assigned == num_nodes);
    // }
}
void __global__ initcost(int* key_person,int* key_item,const double*cost_matrics,double*cost,int cnt,int num_nodes){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >=cnt*cnt) return;
    int i=id/cnt;
    int j=id%cnt;
    cost[id]=cost_matrics[key_person[i]*num_nodes+key_item[j]];
    // printf("%f %d %d %d %d\n",cost[id],num_nodes,key_person[i]*num_nodes+key_item[j],i,key_item[j]);
}
void linear_assignment_auction(
                const double* cost_matrics,
                int*  solutions,
                const int num_graphs,
                const int num_nodes,
                char* scratch,
                char* stop_flags,
                const double auction_max_eps,
                const double auction_min_eps,
                const double auction_factor,
                const int max_iterations)
{
    //get pointers from scratch, size of scratch: num_graphs * (4*num_nodes + num_nodes*num_nodes) * 4 bytes
    int* person2item,*item2person,*sbids,* prices_group;
    double *prices,*bids;
    cudaMalloc(&person2item,num_nodes*sizeof(int));
    cudaMalloc(&item2person,num_nodes*sizeof(int));
    cudaMalloc(&sbids,num_nodes*sizeof(int));
    cudaMalloc(&prices,num_nodes*sizeof(double));
    cudaMalloc(&bids,num_nodes*num_nodes*sizeof(double));


    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    //
    double *s_prices;
    cudaMalloc(&s_prices,num_nodes*sizeof(double));
    //init
    cudaMemsetAsync(prices, 0, num_nodes * sizeof(double));

    //launch solver
    linear_assignment_auction_kernel<double><<<1, 1024>>>
                                    (
                                        num_nodes,
                                        cost_matrics,
                                        person2item,
                                        item2person,
                                        s_prices,
                                        bids,
                                        prices,
                                        // sbids,
                                        stop_flags,
                                        auction_max_eps,
                                        auction_min_eps,
                                        auction_factor,
                                        max_iterations
                                    );
    cudaDeviceSynchronize();
    int *key_item,*key_person;
    cudaMalloc(&key_item,400*sizeof(int));
    cudaMalloc(&key_person,400*sizeof(int));
    int h_key_item[400],h_key_person[400],h_solutions[num_nodes],hh_solutions[num_nodes],cnt=0;
    double *cost;
    cudaMalloc(&cost,400*400*sizeof(double));
    cudaMemcpy(h_solutions, person2item, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hh_solutions, item2person, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0;i<num_nodes;i++){
        if(h_solutions[i]==-1)h_key_person[cnt++]=i;
    }   
    std::cout<<cnt<<'\n';  
    cnt=0;
    for(int i=0;i<num_nodes;i++){
        if(hh_solutions[i]==-1)h_key_item[cnt++]=i;
    }      
    std::cout<<cnt<<'\n';
    cudaMemcpy(key_person, h_key_person, cnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(key_item, h_key_item, cnt * sizeof(int), cudaMemcpyHostToDevice);   
    int blockSize = 256;
    int gridSize = (cnt*cnt + blockSize - 1) / blockSize;     
    initcost<<<gridSize, blockSize>>>(key_person,key_item,cost_matrics,cost,cnt,num_nodes); 
    int* person2item1,*item2person1;
    cudaMalloc(&person2item1,cnt*sizeof(int));
    cudaMalloc(&item2person1,cnt*sizeof(int));
    cudaMemsetAsync(prices, 0, cnt * sizeof(double));
    linear_assignment_auction_kernel<double><<<1, cnt>>>
                                    (
                                        cnt,
                                        cost,
                                        person2item1,
                                        item2person1,
                                        s_prices,
                                        bids,
                                        prices,
                                        // sbids,
                                        stop_flags,
                                        auction_max_eps,
                                        auction_min_eps,
                                        auction_factor,
                                        200
                                    );       
    cudaDeviceSynchronize();
    resetkey<<<1, cnt>>>(person2item1,person2item,key_person,key_item);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "auction calculatekernel time: " << elapsed.count() << " seconds" << std::endl;
    //copy solutions
    cudaMemcpy(solutions, person2item, num_graphs * num_nodes * sizeof(int), cudaMemcpyDeviceToDevice);

}
}
