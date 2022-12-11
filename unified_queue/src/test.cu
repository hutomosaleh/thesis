#include <cooperative_groups.h>
#include <iostream>


/*
  THREAD BLOCK TUTORIAL START
*/

__device__ int reduce_sum(cooperative_groups::thread_group group, int *temp, int val)
{
  // Get index of calling thread in group
  int lane = group.thread_rank();

  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  for (int i = group.size() / 2; i > 0; i /= 2)
  {
    temp[lane] = val; // Store value to temporary memory
    group.sync(); // Wait for all threads to STORE
    if (lane < i) val += temp[lane + i]; // Add partial sum
    group.sync(); // Wait for all threads to LOAD
  }
  return val;
}

__device__ int thread_sum(int *input, int n)
{
  int sum = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 4; i += blockDim.x * gridDim.x)
  {
    int4 in = ((int4*) input)[i];
    sum += in.x + in.y + in.z + in.w;
  }
  return sum;
}

__global__ void sum_kernel_block(int *sum, int *input, int n)
{
  int my_sum = thread_sum(input, n);

  extern __shared__ int temp[];
  auto group = cooperative_groups::this_thread_block();
  int block_sum = reduce_sum(group, temp, my_sum);

  if (group.thread_rank() == 0) atomicAdd(sum, block_sum);
}

/*
  THREAD BLOCK TUTORIAL END
*/

int main(int argc, char *argv[])
{
  // Thread block tutorial - START
  int n = 1<<24;
  int blockSize = 256;
  int nBlocks = (n + blockSize - 1) / blockSize;
  int sharedBytes = blockSize * sizeof(int);

  int *sum, *data;
  cudaMallocManaged(&sum, sizeof(int));
  cudaMallocManaged(&data, n * sizeof(int));
  std::fill_n(data, n, 5); // initialize data
  cudaMemset(sum, 0, sizeof(int));

  sum_kernel_block<<<nBlocks, blockSize, sharedBytes>>>(sum, data, n);
  
  cudaDeviceSynchronize();
  std::cout << "n: " << n << std::endl;
  std::cout << "Data: " << data[0] << std::endl;
  std::cout << "Data: " << data[1] << std::endl;
  // Thread block tutorial - END

  // Partitioned Group - START
//  cooperative_groups::thread_group tile32 = cooperative_groups::partition(cooperative_groups::this_thread_block(), 32);
//  cooperative_groups::thread_group tile4 = cooperative_groups::tiled_partition(tile32, 4);
//  if (tile4.thread_rank() == 0) printf("Hello from tile4 rank 0: %d\n", cooperative_groups::this_thread_block().thread_rank());
  // Partitioned Group - END
  return 1;
}
