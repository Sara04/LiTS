/*
 * lung_segmentator_cuda_kernels.cuh
 *
 *  Created on: Apr 8, 2017
 *      Author: sara
 */

#ifndef LUNG_SEGMENTATOR_CUDA_KERNELS_CUH_
#define LUNG_SEGMENTATOR_CUDA_KERNELS_CUH_

#define MAX_THREADS 1024

/******************************************************************************
 * volume_air_segmentation_gpu: segmentation of the air regions
 *
 * Arguments:
 *      volume: volume with normalized intensities
 *      w: volume width
 *      h: volume height
 *      d: volume depth / number of slices
 *      air_mask: output air mask
 *      threshold: threshold which defines air regions
******************************************************************************/
__global__ void volume_air_segmentation_gpu(const float *volume,
                                            unsigned w, unsigned h, unsigned d,
                                            bool *air_mask, float threshold)
{
    unsigned int idx = blockIdx.y * gridDim.x * blockDim.x
            + blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < (h * w * d))
        air_mask[idx] = (volume[idx] < threshold);
}

/******************************************************************************
 * detect_body_bounds_gpu: detect front, back, left and right body bounds
 * for each slice
 *
 * Arguments:
 *      air_mask: pointer to the array containing air segmentation
 *      bounds: pointer to the array where the body bounds would be
 *          placed
 *      side_threshold: threshold for body side detection
 *      front_threshold: threshold for body front detection
 *      back_threshold: threshold for body back detection
******************************************************************************/
__global__ void detect_body_bounds_gpu(const bool *air_mask, unsigned *bounds,
                                       unsigned w, unsigned h, unsigned d,
                                       unsigned side_threshold,
                                       unsigned front_threshold,
                                       unsigned back_threshold)
{

    __shared__ unsigned int h_sum[MAX_THREADS];
    __shared__ unsigned int v_sum[MAX_THREADS];

    if (threadIdx.x == 0)
    {
        bounds[blockIdx.x] = 0;
        bounds[blockIdx.x + d] = w - 1;
        bounds[blockIdx.x + 2 * d] = 0;
        bounds[blockIdx.x + 3 * d] = h - 1;
    }
    if(threadIdx.x < w)
    {
        float h_c = 0;
        unsigned int v_idx = blockIdx.x * w * h + threadIdx.x;
        for (unsigned int i = 0; i < h; i++)
            h_c += (air_mask[v_idx + i * w] == 0);
        h_sum[threadIdx.x] = h_c;
    }
    if(threadIdx.x < h)
    {
        float v_c = 0;
        unsigned int h_idx = blockIdx.x * w * h + threadIdx.x * w;
        for (unsigned int i = 0; i < w; i++)
            v_c += (air_mask[h_idx + i] == 0);
        v_sum[threadIdx.x] = v_c;
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        unsigned int m = w;
        if(h > m)
            m = h;
        for (unsigned int i = 0; i < m; i++)
        {
            if (i < (w / 2) and i > bounds[blockIdx.x])
                if (h_sum[i] < side_threshold)
                    bounds[blockIdx.x] = i;
            if (i < (h / 2) and i > bounds[blockIdx.x + 2 * d])
                if (v_sum[i] < front_threshold)
                    bounds[blockIdx.x + 2 * gridDim.x] = i;
            if (i >= (w / 2) and i <= bounds[blockIdx.x + d] and i < w)
                if (h_sum[i] < side_threshold)
                    bounds[blockIdx.x + d] = i;
            if (i >= (h / 2) and i <= bounds[blockIdx.x + 3 * d] and i < h)
                if (v_sum[i] < back_threshold)
                    bounds[blockIdx.x + 3 * d] = i;
        }
    }
}



#endif /* LUNG_SEGMENTATOR_CUDA_KERNELS_CUH_ */
