#include <iostream>
#include <fstream>

#include "lung_segmentator.cuh"
#include "../tools/tools_cuda.cuh"
#include "lung_segmentator_cuda_kernels.cuh"

#define MAX_THREADS 1024

/******************************************************************************
 * segment_lungs: segmentation of lungs, calls a number of functions run on
 *  gpu and cpu in order to perform segmentation; it includes detection of
 *  air regions, detection of body bounds, down-scaling, 3d region labeling,
 *  lung candidate extraction based on size and position, segmentation
 *  up-scaling and refining
 *
 * Arguments:
 *      volume: input volume
 *      size: size of the volume
 *      lungs_mask: buffer where lungs segmentation will be stored
 *      ds: down-sampling factor
 *      lung_assumed_center_n: normalized assumed center of the lungs
 *      body_bounds_th: threshold for determining body bounds
 *      lung_v_th: minimal lung size threshold in mm^3
 *      at: air threshold
******************************************************************************/
void segment_lungs(const float *volume, const unsigned int *size,
                   bool *lungs_mask,
                   const unsigned int *ds,
                   float *lung_assumed_center_n,
                   const unsigned int *body_bounds_th,
                   float lung_v_th, float at)
{
    //_______________________________________________________________________//
    // 1. Detecting air regions around and in body
    //_______________________________________________________________________//
    unsigned int volume_l = size[0] * size[1] * size[2];
    unsigned int volume_B = volume_l * sizeof(float);
    float *volume_d;
    bool *air_mask = new bool[volume_l];
    bool *air_mask_d;

    cudaMalloc((void **) &volume_d, volume_B);
    cudaMemcpy(volume_d, volume, volume_B, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &air_mask_d, volume_B);

    unsigned int i1 = size[1];
    unsigned int i2 = volume_l / (size[1] * MAX_THREADS) + 1;
    dim3 grid(i1, i2);
    volume_air_segmentation_gpu<<<grid, MAX_THREADS>>>
            (volume_d, size[0], size[1], size[2], air_mask_d, at);
    cudaFree(volume_d);
    //_______________________________________________________________________//
    // 2. Detecting body bounds
    //_______________________________________________________________________//
    unsigned int *bounds = new unsigned int[4 * size[2]];
    unsigned int *bounds_d;
    cudaMalloc((void **) &bounds_d, 4 * size[2] * sizeof(unsigned int));
    detect_body_bounds_gpu<<<size[2], size[0]>>>
            (air_mask_d, bounds_d, size[0], size[1], size[2],
             body_bounds_th[0], body_bounds_th[1], body_bounds_th[2]);
    cudaMemcpy(bounds, bounds_d, 4 * size[2] * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    cudaFree(bounds_d);
    //_______________________________________________________________________//
    // 3. Down-sample air mask for further processing
    //_______________________________________________________________________//
    unsigned int size_ds[3] = {size[0]/ds[0], size[1]/ds[1], size[2]/ds[2]};
    unsigned int volume_ds_l = size_ds[0] * size_ds[1] * size_ds[2];
    unsigned int air_mask_ds_B = volume_ds_l * sizeof(bool);

    bool *air_mask_ds = new bool[volume_ds_l];
    bool *air_mask_ds_d;

    cudaMalloc((void **) &air_mask_ds_d, air_mask_ds_B);
    dim3 grid_ds(size_ds[1], size_ds[2]);
    subsample_gpu<bool> <<<grid_ds, size_ds[0]>>>
            (air_mask_d, air_mask_ds_d, ds[0], ds[1], ds[2]);
    cudaMemcpy(air_mask_ds, air_mask_ds_d, air_mask_ds_B,
               cudaMemcpyDeviceToHost);
    cudaFree(air_mask_ds_d);
    //_______________________________________________________________________//
    // 4. Remove outside body air
    //_______________________________________________________________________//
    unsigned int *bounds_ds = new unsigned int[4 * size_ds[2]];
    for (unsigned int i = 0; i < 2 * size[2]; i += ds[2])
        bounds_ds[i] = bounds[i] / ds[0];
    for (unsigned int i = 2 * size[2]; i < 4 * size[2]; i += ds[2])
        bounds_ds[i] = bounds[i] / ds[1];
    remove_outside_body_air(air_mask_ds, size_ds, bounds_ds);
    delete[] bounds_ds;
    //_______________________________________________________________________//
    // 5. Determine the center of the largest air object along vertical axis
    //_______________________________________________________________________//
    float lung_center_slice = 0;
    lung_central_slice(air_mask_ds,  size_ds, lung_center_slice);
    lung_assumed_center_n[2] = lung_center_slice / size_ds[2];
    //_______________________________________________________________________//
    // 6. Labeling in body air
    //_______________________________________________________________________//
    unsigned int *labeled = new unsigned int[volume_ds_l];
    unsigned int *object_sizes = new unsigned int[volume_ds_l];
    unsigned int label = 0;
    labeling_3d(const_cast<const bool *>(air_mask_ds),
                labeled, size_ds, object_sizes, label);
    delete[] air_mask_ds;
    //_______________________________________________________________________//
    // 7. Extracting lung candidates from labeled data according to the
    //    size and/or position
    //_______________________________________________________________________//
    bool *candidates = new bool[volume_ds_l];
    float lung_v_th_ds = lung_v_th / (ds[0] * ds[1] * ds[2]);
    extract_lung_candidates(const_cast<const unsigned int *>(labeled),
                            const_cast<const unsigned int *>(size_ds),
                            object_sizes, label, candidates,
                            lung_v_th_ds, lung_assumed_center_n);
    delete[] labeled;
    delete[] object_sizes;
    //_______________________________________________________________________//
    // 8. Up-sample detected mask corresponding to the lungs
    //_______________________________________________________________________//
    bool *candidates_d;
    bool *mask_up_d;
    cudaMalloc((void **) &mask_up_d, volume_l * sizeof(bool));
    cudaMalloc((void **) &candidates_d, volume_ds_l * sizeof(bool));
    cudaMemcpy(candidates_d, candidates, volume_ds_l * sizeof(bool),
               cudaMemcpyHostToDevice);
    delete[] candidates;
    upsample_gpu<bool> <<<grid_ds, size_ds[0]>>>
            (air_mask_d, mask_up_d, candidates_d, ds[0], ds[1], ds[2]);

    cudaMemcpy(air_mask, mask_up_d, volume_l * sizeof(bool),
               cudaMemcpyDeviceToHost);
    cudaFree(mask_up_d);
    cudaFree(candidates_d);
    //_______________________________________________________________________//
    // 9. Re-label up-sampled data
    //_______________________________________________________________________//
    labeled = new unsigned int[volume_l];
    object_sizes = new unsigned int[volume_l];
    label = 0;
    labeling_3d(const_cast<const bool *>(air_mask),
                labeled, size, object_sizes, label);

    extract_lung_candidates(const_cast<const unsigned int *>(labeled),
                            size, object_sizes, label, lungs_mask,
                            lung_v_th, lung_assumed_center_n);
    bool *lungs_mask_d;
    cudaMalloc((void **)&lungs_mask_d,volume_l * sizeof(bool));
    cudaMemcpy(lungs_mask_d, lungs_mask, volume_l * sizeof(bool),
               cudaMemcpyHostToDevice);

    unsigned int up = ds[1];
    if(ds[0] > ds[1])
        up = ds[0];

    dim3 grid_rd(size[1], size[2]);
    for(unsigned int i = 0; i < up; i++)
        refine_detection<bool><<<grid_rd, size[0]>>>(lungs_mask_d, air_mask_d);

    cudaMemcpy(lungs_mask, lungs_mask_d, volume_l * sizeof(bool),
               cudaMemcpyDeviceToHost);
    cudaFree(air_mask_d);
    cudaFree(lungs_mask_d);
    delete[] object_sizes;
    delete[] labeled;
    delete[] air_mask;
    //_______________________________________________________________________//
}

