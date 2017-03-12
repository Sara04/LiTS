#include "liver_detector_cuda.cuh"
#include <iostream>

__global__ void compute_liver_lung_size(const bool *lungs_mask,
                                        const unsigned char *liver_mask,
                                        unsigned int *output,
                                        unsigned int w,
                                        unsigned int h,
                                        unsigned int d)
{

    unsigned int idx = blockIdx.y * gridDim.x * blockDim.x
                       + blockIdx.x * blockDim.x + threadIdx.x;

    bool liver_value = (liver_mask[idx] > 0);
    bool lungs_value = lungs_mask[idx];

    atomicAdd(&output[blockIdx.y], lungs_value);
    atomicAdd(&output[d + blockIdx.x], lungs_value);
    atomicAdd(&output[d + h + threadIdx.x], lungs_value);

    atomicAdd(&output[(w + h + d) + blockIdx.y], liver_value);
    atomicAdd(&output[(w + h + 2 * d) + blockIdx.x], liver_value);
    atomicAdd(&output[(w + 2 * h + 2 * d) + threadIdx.x], liver_value);
}


void estimate_liver_lung_size(const bool *lungs_mask,
                              const unsigned char *liver_mask,
                              const unsigned int *size,
                              const float *voxel_size,
                              const unsigned int *body_bounds,
                              unsigned int *liver_bounds)
{

    unsigned int len = size[0] * size[1] * size[2];
    unsigned int *output = new unsigned int[len * 2];
    for(unsigned int i = 0; i < 2 * len; i++)
        output[i] = 0;

    bool *lungs_mask_gpu;
    unsigned char *liver_mask_gpu;
    unsigned int *output_gpu;

    cudaMalloc((void **)&lungs_mask_gpu, len * sizeof(bool));
    cudaMalloc((void **)&liver_mask_gpu, len * sizeof(unsigned char));
    cudaMalloc((void **)&output_gpu,
               2 * (size[0] + size[1] + size[2]) * sizeof(unsigned int));

    cudaMemcpy(lungs_mask_gpu, lungs_mask, len * sizeof(bool),
               cudaMemcpyHostToDevice);

    cudaMemcpy(liver_mask_gpu, liver_mask, len * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    cudaMemcpy(output_gpu, output,
               2 * (size[0] + size[1] + size[2]) * sizeof(unsigned int),
               cudaMemcpyHostToDevice);

    dim3 grid(size[1], size[2]);

    compute_liver_lung_size<<<grid, size[0]>>>(lungs_mask_gpu,
                                               liver_mask_gpu,
                                               output_gpu,
                                               size[0], size[1], size[2]);
    cudaMemcpy(output, output_gpu,
               2 * (size[0] + size[1] + size[2]) * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);

    cudaFree(lungs_mask_gpu);
    cudaFree(liver_mask_gpu);
    cudaFree(output_gpu);

    unsigned int lungs_b[6] = {0, 0, 0, 0, 0, 0};
    unsigned int liver_b[6] = {0, 0, 0, 0, 0, 0};

    unsigned int n = (size[0] + size[1] + size[2]);
    unsigned int acc = 0;
    for(unsigned int i = 0; i < 3; i++)
    {
        if(i > 0)
            acc += size[(2 - i) + 1];
        for(unsigned int i_idx = 0; i_idx < size[2 - i]; i_idx++)
        {
            if(output[acc + i_idx] and !lungs_b[2 * i])
                lungs_b[2 * i] = i_idx;
            if(output[acc + size[2 - i] - 1 - i_idx] and !lungs_b[2 * i + 1])
                lungs_b[2 * i + 1] = size[2 - i] - 1 - i_idx;

            if(output[n + acc + i_idx] and !liver_b[2 * i])
                liver_b[2 * i] = i_idx;
            if(output[n + acc + size[2 - i] - 1 - i_idx] and !liver_b[2 * i + 1])
                liver_b[2 * i + 1] = size[2 - i] - 1 - i_idx;
        }
    }

    unsigned int lungs_width = lungs_b[5] - lungs_b[4];
    unsigned int lungs_height = lungs_b[3] - lungs_b[2];

    int l_top = 0;

    float mean = 0;
    float count = 0;
    for(unsigned int i_idx = 0; i_idx < size[2]; i_idx++)
    {
        if(output[i_idx] > 0)
        {
            mean += output[i_idx];
            count += 1;
        }
    }
    mean /= count;
    for(unsigned int i_idx = 0; i_idx < size[2]; i_idx++)
    {
        if(output[i_idx] > mean)
        {
            l_top = i_idx;
            break;
        }
        l_top = i_idx;
    }

    unsigned int body_width = body_bounds[lungs_b[0] + size[2]] -
                              body_bounds[lungs_b[0]];
    unsigned int body_height = body_bounds[lungs_b[0] + 3 * size[2]] -
                               body_bounds[lungs_b[0] + 2 * size[2]];
    // Left bound
    liver_bounds[0] = (body_bounds[lungs_b[0]] + lungs_b[4]) / 2;
    if(body_bounds[lungs_b[0]] + int(0.05 * float(body_width)) < liver_bounds[0])
        liver_bounds[0] = body_bounds[lungs_b[0]] + int(0.05 * float(body_width));
    // Right bound
    liver_bounds[1] = (body_bounds[lungs_b[0] + size[2]] + lungs_b[5]) / 2;
    if(body_bounds[lungs_b[0] + size[2]] - int(0.05 * float(body_width)) > liver_bounds[1])
        liver_bounds[1] = body_bounds[lungs_b[0] + size[2]] - int(0.05 * float(body_width));

    // Front bound
    liver_bounds[2] = body_bounds[lungs_b[0] + 2 * size[2]];

    // Back bound
    liver_bounds[3] = lungs_b[3];

    // Top bound
    liver_bounds[5] = lungs_b[0] + 2 * (l_top - lungs_b[0]);
    if (liver_bounds[5] >= size[2])
        liver_bounds[5] = size[2] - 1;

    // Bottom bound
    if((int(liver_bounds[5]) - int(25 * 10 / voxel_size[2])) > 0)
        liver_bounds[4] = liver_bounds[5] - int(25 * 10 / voxel_size[2]);
    else
        liver_bounds[4] = 0;

    delete [] output;
}
