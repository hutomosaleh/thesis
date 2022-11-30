## Relevancy of co-processing in GPU architecture

### Possible thesis topics
- Evaluating co-processing techniques on SPEs
- Investigating GPU accelerated DBMS
- Common database operators with co-processing
- Comparing co-processing with CUDA and OpenCL variants

### Questions
- When is co-processing viable?
  - On large data, where workload should be offloaded to speed up computation time
  - On non-CPU intensive computation, where GPU performs faster than CPU
- What are common co-processing research topic?
  - Algorithm specific comparisons: Group-By, Hash-join, Aggregate, etc
- Can co-processing be done in any systems with GPU?
  - Raspberry Pi? No official support yet. But there are community version for 3B+

### Others
- Common practice seems to use OpenCL instead of CUDA
