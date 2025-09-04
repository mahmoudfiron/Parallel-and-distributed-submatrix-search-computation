## Project Overview
This research project focuses on a **parallel implementation** of submatrix (template) search in grayscale matrices.
Given a set of square **pictures** and a set of square **objects**, the goal is to determine, for each picture, whether **any** object appears inside it under a given **relative-difference threshold**. This is computationally intensive because each object must be slid over all valid positions in the picture and compared window-by-window.

### Problem Definition
For a picture `P∈ZN×N` and an object `O∈Zn×n` with `n≤N`,we examine all top-left positions (i,j) where the object fits: `0≤i,j≤N−n`.

At each position we compute the score:
`score(i,j)=∑r=0,n−1∑c=0,n−1|P[i+r,j+c]−O[r,c]/P[i+r,j+c]|`


If score(i,j)<threshold, we say the object **matches** the picture at (i,j).
Per assignment rules, for each picture we **stop at the first match** (first object/position that satisfies the threshold) and report it; otherwise we report that no objects were found.

### Input and Output Specifications
1. **Input File (`input.txt`)**:
```
<threshold>
<P>                         # number of pictures
<picture_1_id>
<N1>
N1×N1 integers (row-major)...
...
<picture_P_id>
<NP>
NP×NP integers...
<M>                         # number of objects
<object_1_id>
<n1>
n1×n1 integers...
...
<object_M_id>
<nM>
nM×nM integers...
```

   **Example Input:**
```
0.05
1
11
4
10 10 10 10
10 10 10 10
10 10 10 10
10 10 10 10
1
101
2
10 10
10 10
```

2. **Output File (`output.txt`)**:
   - If a match is found:
     ```
     Picture <picId> found Object <objId> in Position(i,j)
     ```
   - If no objects match:
     ```
     Picture <picId> No Objects were found
     ```

## Parallelization Approach
To optimize the computational process, the solution leverages a combination of **MPI**, **OpenMP**, and **CUDA** for efficient parallel computation:

### MPI (Message Passing Interface)
- **Purpose**: Distribute pictures across multiple processes.
- **Rationale**: Pictures are independent; each rank can process a subset.
- **Architecture**: Implement a **Master-Worker Model**:
  - **Rank 0**: reads `input.txt`, then broadcasts the **threshold**, all **pictures**, and all **objects** to all ranks (`MPI_Bcast`).
  - **rank Processes**: Each rank processes picture indices `rank, rank+np, rank+2np, ....`
  - Ranks send local results back to rank 0, which assembles and writes `output.txt`.

### OpenMP (Multi-threading)
- **Purpose**: Parallelize the inner search **within** an MPI rank.
- **Rationale**: For a fixed (picture, object), different candidate rows `i` can be explored concurrently.
- **Construct used**: **OpenMP** tasks. We create **one task per row** `i`, each scanning columns `j`.
A shared atomic **foundFlag** lets the first winning task record `(i,j)` and stop others early (required “task construct” bonus).

### CUDA (GPU Acceleration)
- **Purpose**: Offload the heavy sliding-window comparisons to the GPU.
- **Rationale**: GPUs excel at parallel arithmetic operations, significantly improving performance.
- **Kernel**: one GPU thread per candidate position `(i,j)`; each thread accumulates the relative-difference sum over the `n×n` window and atomically records the first match.
- **Multistreaming**: two CUDA streams (e.g., sCopy and sComp) and **ping-pong buffers**. While the kernel evaluates object k on sComp, we **prefetch object** `k+1` with `cudaMemcpyAsync` on `sCopy` to overlap **H2D transfers** with **compute**.
If no CUDA device exists, the code a**utomatically falls back** to the OpenMP CPU path (same outputs).

### Performance Consideration
- **Expected scaling:**:
  - Increasing OpenMP threads speeds up the per-picture search until memory bandwidth or overheads dominate.
  - MPI improves throughput by distributing pictures across processes/nodes.
  - GPU can outperform CPU significantly for large matrices; multistreaming typically gives an extra boost when H2D copies are non-negligible.

**Expected scaling:**:
   ```
| Setting               | Time (s) | Speedup vs. 1T  |
|-----------------------|---------:|----------------:|
| 1 MPI, OMP=1          |   61.75  | 1.00×           |
| 1 MPI, OMP=4          |   34.11  | 1.81×           |
| 2 MPI, OMP=4          |   31.44  | 1.96×           |
| GPU (single stream)   |   3.09   | 20×             |
| GPU (multistreaming)  |   2.06   | 30×             |
   ```

## How to Run the Project
1. **Build** 
Prerequisites: OpenMPI, GCC with OpenMP; CUDA toolkit (optional for GPU build).
   ```bash
   make clean
   make CC=mpicc USE_CUDA=1      # builds CUDA path; falls back to CPU if no GPU
   ```
   (You can also build CPU-only with `make CC=mpicc`.)
2. **Run locally:**

   ```bash
   export OMP_NUM_THREADS=4
   mpirun -np 2 ./build/pds_project_mpi_omp_c data/input.txt output.txt
   ```

3. Run on SLURM:
   ```bash
   sbatch scripts/run_sbatch.slurm
   ```

## Expected Output
For each picture, either a first match (object id and position) or a “no objects found” line.

**Example Output:**
```
Picture 11 found Object 101 in Position(1,2)
Picture 12 No Objects were found
Picture 13 found Object 102 in Position(2,1)
Picture 14 found Object 103 in Position(1,1)
```

## Implementation Details
- **MPI:** rank 0 parses input; all ranks receive data via `MPI_Bcast`. Work split by picture index. Results gathered at rank 0.
- **OpenMP tasks:** one task per candidate row `i`; each task scans columns `j`. An atomic `foundFlag` enables **early stop** on the first match to avoid wasted work.
- **CUDA:** 
- Kernel maps one thread to one candidate `(i,j)`; the first passing thread uses `atomicCAS` to record `(i,j)`.

- **Multistreaming**: two streams (compute vs. copy) + ping-pong device buffers to overlap transfers of object `k+1` with compute on `k`.

- If `cudaGetDeviceCount()==0`, the GPU routine returns control to the CPU path (identical results).
- **I/O:** simple text format reader/writer; matrices are stored row-major.
- **Code layout:**
```
src/
  main.c           # MPI: broadcast, rank work split, gather, write output
  compute.c        # CPU search (OpenMP tasks, atomic early-stop)
  io.c / io.h      # parsing and output formatting
  types.h          # Picture/Object/MatchResult structs
  cuda_match.cu    # CUDA kernel + multistreaming pipeline (optional)
  cuda_match.h
Makefile
data/
  input.txt, sample_input.txt, bench_hard.txt (large benchmark)
  
```


### Key Equations
- **Matching score** (minimize):
 
 ```
 score(i,j) = Σ_{r=0}^{n-1} Σ_{c=0}^{n-1} [ P[i+r, j+c] * P[i+r, j+c] - O[r,c] ]
 ```

- **Match condition:** score(i,j)<threshold.

***Work per (picture, object):*** `O((N-n+1)^2 * n^2)` in the worst case (no early exit).


### Load Balancing Consideration
- **Across nodes (MPI):** pictures are evenly striped across ranks (`rank, rank+np, ...`) to balance counts even when sizes differ.
- **Within a rank (OpenMP):** using **tasks** gives dynamic distribution of candidate rows; if some rows find a match early, other rows can stop quickly via the shared atomic flag.
- **On GPU:** one thread per `(i,j)` maximizes occupancy; multistreaming overlaps small H2D object copies with kernel compute. For very large objects or many objects, pinned memory (`cudaHostAlloc`) and batched transfers can further improve overlap.