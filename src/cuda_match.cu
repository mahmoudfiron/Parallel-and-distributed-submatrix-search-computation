#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <atomic>
extern "C" {
  #include "types.h"
  #include "cuda_match.h"
}


// Warn only once per MPI process if there is no GPU
static std::atomic<bool> warned{false};

// Device kernel: one thread per (i,j) candidate
__global__ void matchKernel(const int* __restrict__ pic, int N,
                            const int* __restrict__ obj, int n,
                            double threshold,
                            int maxI, int maxJ,
                            int* foundFlag, int* outI, int* outJ)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > maxI || j > maxJ) return;

  // If someone already found, skip
  if (atomicAdd(foundFlag, 0) != 0) return;

  double sum = 0.0;
  for (int r = 0; r < n && sum < threshold; ++r) {
    int baseP = (i + r) * N + j;
    int baseO = r * n;
    for (int c = 0; c < n && sum < threshold; ++c) {
      int pv = pic[baseP + c];
      int ov = obj[baseO + c];
      // per spec pv in [1..100], divide-by-zero not expected
      sum += fabs((double)(pv - ov) / (double)pv);
    }
  }
  if (sum < threshold) {
    if (atomicCAS(foundFlag, 0, 1) == 0) { *outI = i; *outJ = j; }
  }
}

int cuda_find_match_for_picture(const Picture* P,
                                const ObjectT* objects, int M,
                                double threshold,
                                MatchResult* out)
{
  // 0) No GPU? Tell caller to fall back to CPU path.
int devCount = 0;
  cudaError_t err = cudaGetDeviceCount(&devCount);
  if (err != cudaSuccess || devCount == 0) {
    if (!warned.exchange(true, std::memory_order_relaxed)) {
      std::fprintf(stderr, "[CUDA] No CUDA device: falling back to CPU (OpenMP).\n");
    }
    return 0;
  }

  const int N = P->N;
  const size_t picBytes = (size_t)N * (size_t)N * sizeof(int);

  // 1) Allocate/copy the picture once
  int *d_pic = nullptr;
  if (cudaMalloc(&d_pic, picBytes) != cudaSuccess) return 0;
  if (cudaMemcpy(d_pic, P->a, picBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_pic); return 0;
  }

  // 2) Device-side results
  int *d_found=nullptr, *d_outI=nullptr, *d_outJ=nullptr;
  cudaMalloc(&d_found, sizeof(int));
  cudaMalloc(&d_outI, sizeof(int));
  cudaMalloc(&d_outJ, sizeof(int));

  // 3) Create two streams: SCopy (prefetch), SComp (compute)
  cudaStream_t sCopy = nullptr, sComp = nullptr;
  cudaStreamCreate(&sCopy);
  cudaStreamCreate(&sComp);

  // Helper: find next valid object index (n <= N)
  auto next_valid = [&](int start)->int {
    int k = start;
    while (k < M && objects[k].n > N) ++k;
    return k;
  };

  // 4) Prefetch the first valid object (if any)
  int k0 = next_valid(0);
  if (k0 >= M) {
    // No valid objects at all
    cudaStreamDestroy(sCopy); cudaStreamDestroy(sComp);
    cudaFree(d_pic); cudaFree(d_found); cudaFree(d_outI); cudaFree(d_outJ);
    return 0;
  }

  // Ping-pong buffers for objects
  int *d_objA = nullptr, *d_objB = nullptr;
  size_t bytesA = 0, bytesB = 0;
  bool useA = true; // current buffer toggle

  // Prefetch first object into A
  {
    int n = objects[k0].n;
    bytesA = (size_t)n * (size_t)n * sizeof(int);
    cudaMalloc(&d_objA, bytesA);
    cudaMemcpyAsync(d_objA, objects[k0].a, bytesA, cudaMemcpyHostToDevice, sCopy);
  }

  // 5) Main pipeline over valid objects
  for (int k = k0; k < M; /* advanced below */) {
    const ObjectT* O = &objects[k];
    const int n  = O->n;
    const int maxI = N - n, maxJ = N - n;

    // Make sure the prefetched buffer for this k is ready
    cudaStreamSynchronize(sCopy);

    // Choose the ready buffer as "d_obj"
    int* d_obj   = useA ? d_objA : d_objB;

    // Reset device found flag asynchronously on compute stream
    int zero = 0;
    cudaMemcpyAsync(d_found, &zero, sizeof(int), cudaMemcpyHostToDevice, sComp);

    // Launch kernel on compute stream
    const int tilesX = maxJ + 1;
    const int tilesY = maxI + 1;
    dim3 block(16,16);
    dim3 grid((tilesX + block.x - 1) / block.x,
          (tilesY + block.y - 1) / block.y);


          matchKernel<<<grid, block, 0, sComp>>>(d_pic, N, d_obj, n, threshold,
                                       maxI, maxJ, d_found, d_outI, d_outJ);

// (optional) check launch error immediately
cudaError_t kerr = cudaGetLastError();
if (kerr != cudaSuccess) {
    std::fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(kerr));
}


    // While compute runs, prefetch NEXT valid object into the other buffer
    int kNext = next_valid(k + 1);
    if (kNext < M) {
      const int n2 = objects[kNext].n;
      size_t bytes2 = (size_t)n2 * (size_t)n2 * sizeof(int);
      if (useA) {
        if (d_objB) cudaFree(d_objB);
        bytesB = bytes2;
        cudaMalloc(&d_objB, bytesB);
        cudaMemcpyAsync(d_objB, objects[kNext].a, bytesB, cudaMemcpyHostToDevice, sCopy);
      } else {
        if (d_objA) cudaFree(d_objA);
        bytesA = bytes2;
        cudaMalloc(&d_objA, bytesA);
        cudaMemcpyAsync(d_objA, objects[kNext].a, bytesA, cudaMemcpyHostToDevice, sCopy);
      }
    }

    // Wait for the kernel to finish
    cudaStreamSynchronize(sComp);

    // Check whether we found a match for this object
    int h_found = 0;
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_found) {
      int i, j;
      cudaMemcpy(&i, d_outI, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&j, d_outJ, sizeof(int), cudaMemcpyDeviceToHost);

      out->pictureId = P->id;
      out->found     = 1;
      out->objectId  = O->id;
      out->posI      = i;
      out->posJ      = j;

      // Cleanup
      if (d_objA) cudaFree(d_objA);
      if (d_objB) cudaFree(d_objB);
      cudaStreamDestroy(sCopy);
      cudaStreamDestroy(sComp);
      cudaFree(d_pic); cudaFree(d_found); cudaFree(d_outI); cudaFree(d_outJ);
      return 1;
    }

    // No match: advance to next valid object
    if (kNext >= M) {
      // No more prefetches pending; we’ll drop out after loop
      // free the current buffer we just used
      if (useA) { cudaFree(d_objA); d_objA = nullptr; bytesA = 0; }
      else      { cudaFree(d_objB); d_objB = nullptr; bytesB = 0; }
      k = kNext; // == M, end loop
    } else {
      // Free the current buffer; swap to the newly prefetched one next iter
      if (useA) { cudaFree(d_objA); d_objA = nullptr; bytesA = 0; }
      else      { cudaFree(d_objB); d_objB = nullptr; bytesB = 0; }
      useA = !useA; // toggle buffer (the "other" one has the next object)
      k = kNext;
    }
  }

  // If we got here: no object matched on GPU
  if (d_objA) cudaFree(d_objA);
  if (d_objB) cudaFree(d_objB);
  cudaStreamDestroy(sCopy);
  cudaStreamDestroy(sComp);
  cudaFree(d_pic); cudaFree(d_found); cudaFree(d_outI); cudaFree(d_outJ);
  return 0;
}

