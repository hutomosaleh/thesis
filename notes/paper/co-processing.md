## Accelerating Group-By and Aggregation on Heterogenous CPU/GPU Platforms

Motivation: integrated GPU has been around and can bypass the PCI-e bottleneck
Goal: Co-process group-by and aggregation algorithm

Setup
- Hardware: Intel Core i7-8700 + UHD Graphics 630
- Tested on 2M and 32M tuples
- Uses uniform and zipf distribution
- Initial hash table is done in CPU
- Optimal cpu:gpu ratio is based on previous executions
- Has 3 major kernels: `build_hash_table`, `gather_item_num`, `aggregate_result`

Co-processing Algorithm
- Create data structure
- Calculate tuples amount and ratio
- Run `build_hash_table` building hash table (both)
- Wait until done
- Run `gather_item_num` and `aggregate_result` in CPU

Observation
- Integrated GPU is "weaker" than CPU, but performs better in most cases
- Uniform results:
  - Achieves up to 2.3x speedup compared to full CPU
    - On smaller dataset, full-CPU is faster, due to computation faster in CPU
      - Offloading to GPU makes sense if CPU is busy / overloaded
  - Always faster (up to 1.7x) than full GPU
- Skewed results:
  - GPU performs much better on skewed results
  - Co-processing achieves up to 6.6x speedup over the full CPU variant

## Towards GPUs being mainstream in analytical processing

Motivation: dGPU are uncommon on modern database system and iGPU is promising
Goal: Show reduced overhead of using GPU and accelerate performance

Introduction:
- Programming an iGPU is simplified nowadays
- iGPU kernels can be run after allocating memory in CPU

Setup
- Hardware: AMD A10-7850K (APU) | Radeon HD 7970 (dGPU)
- Uses well-known CPU scan algorithm: BitWeaving
- Adjustment to BitWeaving is done for GPU kernels
- OpenCL flow
  - Copy data (DMA)
  - Launch scan (C)
  - Execute scan (G)
  - Launch aggregate (C)
  - Execute aggregate (G)
  - Check done (C)
  - Copy result (DMA)
  - Return result (C)
- A significant time is spent on aggregation (~80%)
  - We offload aggregate computation to GPU
- To allow parallel execution, synchronization to guard the reduction variables are used
  - lock-free algorithms and local reduction tree

Observations
- Current implementation requires many OS calls, which can induce overhead on dGPU
- Scan results
  - iGPU performs 3x faster and consumes 3x less energy than dGPU
  - iGPU performs slightly faster than multicore CPU ~17% speed up and ~27% less energy consumed
- TPC-H results
  - For some queries, iGPU outperforms multicore CPU
  - Overall iGPU increases performance compared to scan and aggregate portion of TPC-H queries
- Aggregate results
  - Simple reduction and large groups performs better on iGPU
  - On small groups, CPU outperforms iGPU
    - Due to memory overhead for each single GPU thread
    - Per-thread memory is not optimized on the GPU

Conclusion: iGPU is a viable way to accelerate query computations and is a great alternative for dGPU

## Revisiting Co-Processing for Hash Joins on the Coupled CPU-GPU Architecture

Motivation: Low PCI-e bandwidth is a bottleneck for Co-processing, iGPU a solution for it 
Goal: Implement GPU accelerated hash-join algorithm on coupled architecture

Setup
- Hardware: AMD Fusion A8-3870K (APU) | Radeon HD 7970 (dGPU)
- Experiment on simple and partitioned hash joins
- Cases: Offloading, Data Dividing, Pipelined Execution
  - PL: We look into the workload and assigned to appropriate processor
  - DD: Simply assigns to processor regardless of workload type
  - OL: Assigns computation on only one processor type
- Uses cost model to parametrize the performance > They call it the abstraction model
- They also investigate overhead of data transfer on dGPU
- They evaluate accuracy of their cost model

Conclusion: Improved performance by 53%, 35%, and 28% over CPU, GPU, and conventional co-processing

Notes
- To recreate, I need to master hash join algorithm and OpenCL programming

## GPU Processing of Theta-Join

Motivation: Theta join is notoriously slow
Problem: GPU implementation of theta join differs from hash- and sort-based equality joins
Goal: Provide GPU implementation of theta join and optimization

Setup
- Hardware: Intel i7-5820K | GTX 970
- Uses CUDA Framework
- Uses [MapReduce](https://en.wikipedia.org/wiki/MapReduce) environment
- Two types of queries
  - Theta join followed by aggregation (Input bound, output size is small)
  - Generic theta join (Output bound, focuses on writing the result)

Observation
- Each case has its own specific issues and importance, which calls for specialized solutions

Conclusion: An order of magnitude improvement over naive implementation. Up to 2.7x on second case. 

Notes
- Theta join is inequality join
- Theta join is an inner join operation which combines tuples on certain condition

