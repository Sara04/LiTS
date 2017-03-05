/*
 * tools_cuda.cuh
 *
 *  Created on: Feb 25, 2017
 *      Author: sara
 */

#ifndef TOOLS_CUDA_CUH_
#define TOOLS_CUDA_CUH_

template<typename T>
__global__ void subsample_gpu(T *volume, T *sub_volume, unsigned int step0,
                              unsigned int step1, unsigned int step2)
{

    unsigned int in_idx = (step2 * blockIdx.y * step1 * gridDim.x * step0
                           * blockDim.x
                           + step1 * blockIdx.x * step0 * blockDim.x
                           + step0 * threadIdx.x);
    unsigned int out_idx = blockIdx.y * gridDim.x * blockDim.x
            + blockIdx.x * blockDim.x + threadIdx.x;
    sub_volume[out_idx] = volume[in_idx];
}

template<typename T>
__global__ void upsample_gpu(T *volume, T *sub_volume, unsigned int step0,
                             unsigned int step1, unsigned int step2)
{
    unsigned int in_idx = blockIdx.y * gridDim.x * blockDim.x
            + blockIdx.x * blockDim.x + threadIdx.x;

    if (!sub_volume[in_idx])
    {
        unsigned int out_idx;

        for (unsigned int s_idx = 0; s_idx < step2; s_idx++)
        {
            for (unsigned int r_idx = 0; r_idx < step1; r_idx++)
            {
                for (unsigned int c_idx = 0; c_idx < step0; c_idx++)
                {
                    out_idx = (step2 * blockIdx.y + s_idx) * step1 * gridDim.x
                              * step0 * blockDim.x
                              + (step1 * blockIdx.x + r_idx) * step0
                                * blockDim.x
                              + (step0 * threadIdx.x + c_idx);
                    volume[out_idx] = false;
                }
            }
        }
    }
}

#endif /* TOOLS_CUDA_CUH_ */
