## Accelerating Group-By and Aggregation on Heterogenous CPU/GPU Platforms

### Summary

Motivation: integrated GPU has been around and can bypass the PCI-e bottleneck
Goal: Co-process group-by and aggregation algorithm

Setup
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

### Summary

Motivation: dGPU are uncommon on modern database system and iGPU is promising
Goal: Show reduced overhead of using GPU and accelerate performance

Introduction:
- Programming an iGPU is simplified nowadays
- iGPU kernels can be run after allocating memory in CPU

Setup
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
