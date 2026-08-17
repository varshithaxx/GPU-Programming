#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__device__ void gridBarrier(int *count, int *generation, int numBlocks)
{
    // Synchronize all threads within the block
    __syncthreads();

    // Only thread 0 of each block updates the counter
    if (threadIdx.x == 0)
    {
        int myGeneration = *generation;

        // Count this block's arrival
        int arrived = atomicAdd(count, 1) + 1;

        // Last block to arrive
        if (arrived == numBlocks)
        {
            *count = 0;

            // Make previous memory writes visible
            __threadfence();

            // Release all waiting blocks
            atomicAdd(generation, 1);
        }
        else
        {
            // Wait until the last block arrives
            while (*((volatile int *)generation) == myGeneration)
            {
            }
        }
    }

    // Synchronize all threads within the block again
    __syncthreads();
}

__global__ void barrierKernel(int *data,
                              int *result,
                              int *count,
                              int *generation,
                              int numBlocks)
{
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;

    // Phase 1
    if (threadId == 0)
    {
        data[blockId] = blockId * 10;
    }

    __syncthreads();

    // Grid-wide barrier
    gridBarrier(count, generation, numBlocks);

    // Phase 2
    if (threadId == 0)
    {
        int previousBlock;

        if (blockId == 0)
            previousBlock = numBlocks - 1;
        else
            previousBlock = blockId - 1;

        result[blockId] = data[previousBlock];
    }
}

int main()
{
    int numBlocks = 8;
    int threadsPerBlock = 32;

    int *d_data;
    int *d_result;
    int *d_count;
    int *d_generation;

    // Allocate GPU memory
    cudaMalloc(&d_data, numBlocks * sizeof(int));
    cudaMalloc(&d_result, numBlocks * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_generation, sizeof(int));

    // Initialize counter and generation
    cudaMemset(d_count, 0, sizeof(int));
    cudaMemset(d_generation, 0, sizeof(int));

    // Get GPU properties
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Check how many blocks can be active simultaneously
    int maxActiveBlocksPerSM;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM,
        barrierKernel,
        threadsPerBlock,
        0
    );

    int maxConcurrentBlocks =
        maxActiveBlocksPerSM * prop.multiProcessorCount;

    if (numBlocks > maxConcurrentBlocks)
    {
        cout << "Cannot safely run the kernel." << endl;
        cout << "Maximum concurrent blocks: "
             << maxConcurrentBlocks << endl;

        return 1;
    }

    // Launch kernel
    barrierKernel<<<numBlocks, threadsPerBlock>>>(
        d_data,
        d_result,
        d_count,
        d_generation,
        numBlocks
    );

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy results from GPU to CPU
    int h_result[8];

    cudaMemcpy(
        h_result,
        d_result,
        numBlocks * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    // Display results
    cout << "Results:" << endl;

    for (int i = 0; i < numBlocks; i++)
    {
        cout << "Block " << i
             << " -> result = "
             << h_result[i] << endl;
    }

    // Free GPU memory
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_count);
    cudaFree(d_generation);

    return 0;
}