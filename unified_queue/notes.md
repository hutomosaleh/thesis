# Observation
- cudaMallocManaged is much slower than cudaMalloc
- Task size influences the time result by a lot
  - 20000 is great for GPU, but slow on CPU
  - 10000 is good for both GPU and CPU
  - Smaller task size gives large overhead on GPU but not in CPU

# TODOs
- Check out Rez's implementation
  - How is the task chunk
    - Chunk consists of dataArray of tuple values
- Implement CPU/GPU kernels into tasks
- Implement task finish callback
- Implement task manager
  - Implement threads and GPU states
  - Implement locks on thread and GPU states
- Implement task generator
  - Implement queue of tasks

Questions:
- How to know if process is idle?
- What data type is a task queue?
- What data type is a task?
- How big is the data chunk?
- How to save data inside Task Class?
  - The result is whats important

Concern:
- Is checking process state even possible? Yes, with self implementation
- How much overhead for task queue and process state check? Need to check our own

# Structure

## main
- initialize data set
- run task generator
- run task manager 

## task generator
- input: raw dataset, chunk size
- generate task class
- generate task queue
- return queue

## task manager
- spawn two threads for CPU and GPU
- each thread consumes tasks and appends results

## task
- tuple: double, double, double, int : 28 Byte
- each task has a size
- holds data and query
- consists of CPU or GPU function (Hard coded)

# Ideas

## How to organize data chunks for processing
- Use variable data chunk

## How to check if task is done
- Use variable to indicate if task is done
- Use callbacks for notifying if task is done

## How to check if GPU/CPU is idle
- If task in GPU is done, then it is idle and ready to run again
- CPU
  - Keep track of thread counts
  - If task is done, mark thread as idle
