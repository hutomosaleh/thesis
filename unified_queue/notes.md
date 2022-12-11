Questions:
- How to know if process is idle?
- What data type is a task queue?
- What data type is a task?

Ideas:
- Tuple number is task number

Concern:
- Is checking process state even possible?
- How much overhead for task queue and process state check?

## Structure:
- main
  - initialize data set
  - run task generator
  - run task scheduler
- task generator
  - input: raw dataset
  - generate task class
  - generate task queue
  - return queue
- task
  - has state variable
  - consists of CPU or GPU function
  - callbacks are called after function finishes

# Ideas

## How to check if task is done
- Use variable to indicate if task is done
- Use callbacks for notifying if task is done

## How to check if GPU/CPU is idle
- If task in GPU is done, then it is idle and ready to run again
- CPU
  - Keep track of thread counts
  - If task is done, mark thread as idle
