- Multiple query languages: C++, Java, Python, SQL, PSL
- APIs: Stream processing, complex event processing, spatial data processing, linear algebra
- Query example
  - Wind turbine + solar panel > union > window over a time + customer > join > mqtt sink

### Under the hood

#### Structure
NES Coordinator
  - Query Optimizer
  - Query & Source catalog
  - Topology Manager
  - RPC System
NES Worker
  - Query Compiler
  - State Management
  - Stateful operator
  - Sensor Management
  - Task-Based Dataflow Engine

#### How to merge queries?
There is a query Optimizer:
- Query Rewrite
- Signature Computation
- Sharing Identification
- Global Query Update
- Global Query Plan
- Query Placement

Steps:
- Signature capturing semantic information of a query
- Identifying similarities among running and newly arriving queries
- Managing shared query plans for thousands of queries

Why merge queries? Throughput is greatly increased! It is crucial to support concurrent queries!

#### Where to place individual operator?

Challenge: We have to map thousands of queries on millions of nodes.

There is a operator placement strategy:
- Constructive Placement
  - Bottom Up
  - Top Down
- Cost-Based Placement
  - Random Search
  - Integer Linear Programming
  - Genetic Algorithm
It combines both strategy.

#### How to efficiently leverage the individual nodes? 

Problem: Current SPEs use hardware resources inefficiently
- Why is it like that?
  - Because they use interpretation-based processing model
  - Upfront-partitioning causes high overhead
  - SPEs do not react to changing data-characteristics

Solution: Use adaptive Grizzly's Query Compilation Strategy
1. Query Optimization and Compilation
  1. Central Optimizer
  2. Local Optimizer
  3. Pipelined Execution Plan
  4. Query Optimizer
  5. Adaptive Execution Engine (Grizzly)
2. Task-Based Execution
3. Profile and re-optimize (Loop back to 1st step)

Grizzly:
- We fuse --> Optimize --> Generate
- Input Stream will be go through two pipelines to improve performance

#### Others

Outlook: Supporting polyglot workloads (i.e. SQL with Javascript and Python functions)
  - Problem: Bottleneck on runtime invocation, data exchange and data conversion
  - Solution: Polyglot query execution with Babelfish
Outlook: Integrating Large Operator State with Rhino

