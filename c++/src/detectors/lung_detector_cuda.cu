#include "lung_detector_cuda.cuh"
#include "../tools/tools_cuda.cuh"
#include "../tools/tools.h"
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
 * volume_air_segmentation_gpu: segmentation of the air regions
 *
 * Arguments:
 * 		volume: pointer to the array with normalized voxel intensities
 * 		air_mask: pointer to the bool air where segmentation results
 * 			would be placed
 * 		threshold: value below which voxel intensities are considered
 * 			to belong to the air
 */
__global__ void volume_air_segmentation_gpu(const float *volume,
                                            bool *air_mask,
                                            float threshold)
{
    unsigned int idx = blockIdx.y * gridDim.x * blockDim.x
            + blockIdx.x * blockDim.x + threadIdx.x;

    air_mask[idx] = (volume[idx] < threshold);
}

/*
 * detect_body_bounds_gpu: detect front, back, left and right
 * 		body bounds for each slice
 *
 * Arguments:
 * 		air_mask: pointer to the array containing air segmentation
 * 		bounds: pointer to the array where the body bounds would be
 * 			placed
 * 		side_threshold: threshold for body side detection
 * 		front_threshold: threshold for body front detection
 * 		back_threshold: threshold for body back detection
 */
__global__ void detect_body_bounds_gpu(const bool *air_mask,
                                       unsigned int *bounds,
                                       unsigned int side_threshold,
                                       unsigned int front_threshold,
                                       unsigned int back_threshold)
{

    __shared__ unsigned int h_sum[512];
    __shared__ unsigned int v_sum[512];

    if (threadIdx.x == 0)
    {
        bounds[blockIdx.x] = 0;
        bounds[blockIdx.x + gridDim.x] = blockDim.x - 1;
        bounds[blockIdx.x + 2 * gridDim.x] = 0;
        bounds[blockIdx.x + 3 * gridDim.x] = blockDim.x - 1;
    }

    float h_c = 0;
    float v_c = 0;

    unsigned int h_idx = blockIdx.x * blockDim.x * blockDim.x
            + threadIdx.x * blockDim.x;
    unsigned int v_idx = blockIdx.x * blockDim.x * blockDim.x + threadIdx.x;

    for (unsigned int i = 0; i < blockDim.x; i++)
    {
        v_c += (air_mask[h_idx + i] == 0);
        h_c += (air_mask[v_idx + i * blockDim.x] == 0);
    }

    h_sum[threadIdx.x] = h_c;
    v_sum[threadIdx.x] = v_c;

    __syncthreads();

    if (threadIdx.x == 0)
    {
        for (unsigned int i = 0; i < blockDim.x; i++)
        {

            if (i < (blockDim.x / 2))
            {
                if (i > bounds[blockIdx.x])
                {
                    if (h_sum[i] < side_threshold)
                        bounds[blockIdx.x] = i;
                }
                if (i > bounds[blockIdx.x + 2 * gridDim.x])
                {
                    if (v_sum[i] < front_threshold)
                        bounds[blockIdx.x + 2 * gridDim.x] = i;
                }
            }
            if (i >= (blockDim.x / 2))
            {
                if (i <= bounds[blockIdx.x + gridDim.x])
                {
                    if (h_sum[i] < side_threshold)
                        bounds[blockIdx.x + gridDim.x] = i;
                }
                if (i <= bounds[blockIdx.x + 3 * gridDim.x])
                {
                    if (v_sum[i] < back_threshold)
                        bounds[blockIdx.x + 3 * gridDim.x] = i;
                }
            }
        }
    }
}

/*
 * is_in_body_box: verify if pixel coordinates are within
 * 		body bounds of the given slice
 *
 * Arguments:
 * 		slice_bounds: side, front and back body bounds
 * 		r_idx: pixel's row
 * 		c_idx: pixel's column
 */
bool is_in_body_box(const unsigned int *slice_bounds, unsigned int c_idx,
                    unsigned int r_idx)
{
    if (slice_bounds[0] <= c_idx and c_idx <= slice_bounds[1]
        and slice_bounds[2] <= r_idx and r_idx <= slice_bounds[3])
        return true;
    return false;
}

/*
 * remove_outside_body_air: remove air segments that are outside
 * 		body, and for sure do not belong to lungs
 *
 * Arguments:
 * 		air_mask: pointer to the air segmentation from which the outside
 * 			body segments should be removed
 * 		size: pointer to the array containing air mask size
 * 		bounds: pointer to the array containing side, front and back bounds
 */
void remove_outside_body_air(bool *air_mask, const unsigned int *size,
                             const unsigned int *bounds)
{
    unsigned int s_bounds[4];
    for (unsigned int s_idx = 0; s_idx < size[2]; s_idx++)
    {
        for (unsigned i = 0; i < 4; i++)
            s_bounds[i] = bounds[s_idx + i * size[2]];

        for (unsigned int r_idx = 0; r_idx < size[0]; r_idx++)
        {
            for (unsigned int c_idx = 0; c_idx < size[1]; c_idx++)
            {
                if (!is_in_body_box(s_bounds, c_idx, r_idx))
                {
                    air_mask[s_idx * size[0] * size[1] +
                             r_idx * size[0] + c_idx] = false;
                }
                else
                {
                    if ((r_idx == s_bounds[2] or r_idx == s_bounds[3]) and
                        (c_idx >= s_bounds[0] and c_idx <= s_bounds[1]))
                    {
                        if (air_mask[s_idx * size[0] * size[1] +
                                     r_idx * size[0] + c_idx])
                            region_growing_2d(&air_mask[s_idx *
                                                        size[0] *
                                                        size[1]],
                                              size, r_idx, c_idx);
                    }
                    if ((c_idx == s_bounds[0] or c_idx == s_bounds[1]) and
                        (r_idx >= s_bounds[2] and r_idx <= s_bounds[3]))
                    {
                        if (air_mask[s_idx * size[0] * size[1] + r_idx * size[0]
                                     + c_idx])
                            region_growing_2d(&air_mask[s_idx *
                                                        size[0] *
                                                        size[1]],
                                              size, r_idx, c_idx);
                    }
                }
            }
        }
    }
}

/*
 * extract_lung_labels: extract segmented objects from the array labeled
 * 		for the first count labels from the main_labels array
 *
 * Arguments:
 * 		labeled: pointer to the array containing labeled within body air
 * 			segments
 * 		candidates: pointer to the array where segments with desired
 * 			labels would be placed
 * 		size: pointer to the array containing size of the labeled array
 * 		main_labels: pointer to the array containing object labels
 * 		count: number of labels from the array main_labels to be
 * 			extracted from the labeled array
 */
void extract_lung_labels(const unsigned int *labeled, bool *candidates,
                         const unsigned int *size,
                         const unsigned int *main_labels, unsigned int count)
{
    unsigned int len = size[0] * size[1] * size[2];
    for (unsigned int i = 0; i < len; i++)
    {
        for (unsigned int j = 0; j < count; j++)
        {
            if (labeled[i] == main_labels[j])
            {
                candidates[i] = true;
                break;
            }
            candidates[i] = false;
        }
    }
}

/*
 * extract_lung_candidates: extract binary mask that covers both lung wings
 *
 * Arguments:
 * 		labeled: pointer to the array containing labeled within body
 * 			air segments
 * 		size: pointer to the array containing size of the labeled array
 * 		object_sizes: pointer to the array where the size of the segmented
 * 			objects are placed
 * 		label: reference to the number of labels in the labeled array
 * 		candidates: pointer to the array where the mask corresponding to the
 * 			lung wings would be placed
 * 		size_threshold: reference to the estimated lung size threshold
 * 		lung_assumed_c_n: assumed normalized center of the lungs
 */
void extract_lung_candidates(const unsigned int *labeled,
                             const unsigned int *size,
                             unsigned int *object_sizes, unsigned int &label,
                             bool *candidates, float &size_threshold,
                             const float *lung_assumed_c_n)
{

    // 1. Count the number of labeled segments whose size is not negligible
    //_______________________________________________________________________//
    unsigned int count = 0;
    unsigned int ng_f = 40;		// negligible segment size factor
    unsigned int s_f = 200;      // segment size comparison factor
    for (unsigned int i = 1; i < label; i++)
        count += (object_sizes[i - 1] > (size_threshold / ng_f));
    //_______________________________________________________________________//

    // 2. Sort main segment candidates according to their size
    //_______________________________________________________________________//
    unsigned int *main_candidates = new unsigned int[count];
    unsigned int *main_labels = new unsigned int[count];
    unsigned int p;

    count = 0;
    for (unsigned int i = 1; i < label; i++)
    {
        if (object_sizes[i - 1] > (size_threshold / ng_f))
        {
            p = count;
            for (unsigned int j = 0; j < count; j++)
            {
                if (object_sizes[i - 1] > main_candidates[j])
                {
                    for (unsigned int k = count; k > j; k--)
                    {
                        main_candidates[k] = main_candidates[k - 1];
                        main_labels[k] = main_labels[k - 1];
                    }
                    p = j;
                    break;
                }
            }
            main_candidates[p] = object_sizes[i - 1];
            main_labels[p] = i;
            count += 1;
        }
    }
    //_______________________________________________________________________//

    // 3. Extract binary mask corresponding to the lungs
    //_______________________________________________________________________//

    // 3.1. If there is only one object, consider it as the mask object
    //      belonging to the both lungs
    //_______________________________________________________________________//
    if (count == 1)
    {
        extract_lung_labels(labeled, candidates, size, main_labels, 1);
        goto end_detection1;
    }
    //_______________________________________________________________________//

    // 3.2. If the largest object is far larger than the second largest one,
    //      consider it as the object covering both lungs
    //_______________________________________________________________________//
    else if ((main_candidates[0] / main_candidates[1]) > s_f)
    {
        extract_lung_labels(labeled, candidates, size, main_labels, 1);
        goto end_detection1;
    }
    //_______________________________________________________________________//

    // 3.3. In addition to the segment size, include information about
    //      segment position into decision making
    //_______________________________________________________________________//
    else
    {
        // 3.3.1. Determine normalized centers of mass for each segment whose
        //        size is not negligible
        //___________________________________________________________________//
        float *central_c = new float[count];
        float *central_r = new float[count];
        float *central_s = new float[count];
        for (unsigned int i = 0; i < count; i++)
        {
            central_c[i] = 0.0;
            central_r[i] = 0.0;
            central_s[i] = 0.0;
            center_of_mass(labeled, size, main_labels[i], central_c[i],
                           central_r[i], central_s[i]);
        }
        //___________________________________________________________________//

        // 3.3.2. If the largest segment is well centered and larger enough
        //        then the second largest segment, it is considered as the
        //        segment belonging to the both lung wings
        //___________________________________________________________________//

        float center_c_th = 0.4;     // normalized column center threshold
        float center_r_th = 0.3;     // normalized row center threshold
        float slice_th = 0.5;        // normalized slice threshold
        float size_f = 2.0;          // slice size factor

        if (central_c[0] < (1. - center_c_th) and central_c[0] > center_c_th
            and central_r[0] < (1. - center_r_th) and central_r[0] > center_r_th
            and central_s[0] > slice_th
            and (main_candidates[0] / main_candidates[1]) > size_f)
        {
            extract_lung_labels(labeled, candidates, size, main_labels, 1);
            goto end_detection2;
        }
        //___________________________________________________________________//

        // 3.3.3. Verify if there is a pair of segments covering one each
        //        lung separately, or if that is not case find the segment
        //        that is closest to the assumed center
        //___________________________________________________________________//
        else
        {

            // 3.3.3.1 Verify if there is a pair of segments that most likely
            //         correspond to the lung wings
            //_______________________________________________________________//
            float center_c_th = 0.4;    // normalized column center threshold
            float center_r_d = 0.1;     // normalized row center distance
                                        // threshold
            float center_s_d = 0.1;     // normalized slice center distance
                                        // threshold

            for (unsigned int i = 0; i < (count - 1); i++)
            {
                for (unsigned int j = i + 1; j < count; j++)
                {
                    if (((central_c[i] < center_c_th and central_c[j]
                            > (1. - center_c_th))
                         or (central_c[j] < center_c_th and central_c[i]
                                 > (1. - center_c_th)))
                        and abs(central_r[i] - central_r[j]) < center_r_d
                        and abs(central_s[i] - central_s[j]) < center_s_d)
                    {
                        main_labels[0] = main_labels[i];
                        main_labels[1] = main_labels[j];
                        extract_lung_labels(labeled, candidates, size,
                                            main_labels, 2);
                        goto end_detection2;
                    }
                }
            }
            //_______________________________________________________________//

            // 3.3.3.2 Find the segment that is closest to the assumed center
            //_______________________________________________________________//
            float min_dist = 1.0;
            float current_dist = 0.0;
            unsigned int m = 0;
            for (unsigned int i = 0; i < count; i++)
            {
                current_dist = sqrt(
                        pow(central_c[i] - lung_assumed_c_n[0], 2) +
                        pow(central_r[i] - lung_assumed_c_n[1], 2) +
                        pow(central_s[i] - lung_assumed_c_n[2], 2));
                if (current_dist < min_dist)
                {
                    min_dist = current_dist;
                    m = i;
                }
            }
            main_labels[0] = main_labels[m];
            extract_lung_labels(labeled, candidates, size, main_labels, 1);
            goto end_detection2;
            //_______________________________________________________________//
        }

        // 3.3.4. Release memory from arrays containing position info
        //___________________________________________________________________//
        end_detection2:
        {
            delete[] central_c;
            delete[] central_r;
            delete[] central_s;
            goto end_detection1;
        }
    }
    //_______________________________________________________________________//

    // 4. Release memory containing size and label info
    //_______________________________________________________________________//

    end_detection1:
    {
        delete[] main_candidates;
        delete[] main_labels;
        return;
    }
    //_______________________________________________________________________//
}

void segment_lungs(const float *volume, const unsigned int *volume_s,
                   bool *lungs_mask,
                   const unsigned int *subsample_f,
                   const float *lung_assumed_center_n,
                   const unsigned int *body_bounds_th,
                   float lung_volume_threshold, float air_threshold)
{
    // 1. Detecting air regions around and in body
    //_______________________________________________________________________//
    unsigned int volume_l = volume_s[0] * volume_s[1] * volume_s[2];
    float *volume_gpu;
    bool *air_mask = new bool[volume_l];
    bool *air_mask_gpu;

    cudaMalloc((void **) &volume_gpu, volume_l * sizeof(float));
    cudaMalloc((void **) &air_mask_gpu, volume_l * sizeof(bool));
    cudaMemcpy(volume_gpu, volume, volume_l * sizeof(float),
               cudaMemcpyHostToDevice);

    dim3 grid(volume_s[1], volume_s[2]);
    volume_air_segmentation_gpu<<<grid, volume_s[0]>>>(volume_gpu,
            air_mask_gpu,
            air_threshold);
    cudaFree(volume_gpu);
    //_______________________________________________________________________//

    // 2. Detecting body bounds
    //_______________________________________________________________________//
    unsigned int *bounds = new unsigned int[4 * volume_s[2]];
    unsigned int *bounds_gpu;

    cudaMalloc((void **) &bounds_gpu, 4 * volume_s[2] * sizeof(unsigned int));
    detect_body_bounds_gpu<<<volume_s[2], volume_s[0]>>>(air_mask_gpu,
            bounds_gpu,
            body_bounds_th[0],
            body_bounds_th[1],
            body_bounds_th[2]);
    cudaMemcpy(bounds, bounds_gpu, 4 * volume_s[2] * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);
    cudaFree(bounds_gpu);
    //_______________________________________________________________________//

    // 3. Sub-sample air mask for further processing
    //_______________________________________________________________________//

    unsigned int s_volume_s[3] = {volume_s[0] / subsample_f[0], volume_s[1]
            / subsample_f[1],
                                  volume_s[2] / subsample_f[2]};
    unsigned int s_volume_l = s_volume_s[0] * s_volume_s[1] * s_volume_s[2];

    bool *air_mask_sub = new bool[s_volume_l];
    bool *air_mask_sub_gpu;

    cudaMalloc((void **) &air_mask_sub_gpu, s_volume_l * sizeof(bool));
    dim3 grid_sub(s_volume_s[1], s_volume_s[2]);
    subsample_gpu<bool> <<<grid_sub, s_volume_s[0]>>>(air_mask_gpu,
            air_mask_sub_gpu,
            subsample_f[0],
            subsample_f[1],
            subsample_f[2]);
    cudaMemcpy(air_mask_sub, air_mask_sub_gpu, s_volume_l * sizeof(bool),
               cudaMemcpyDeviceToHost);
    cudaFree(air_mask_sub_gpu);

    //_______________________________________________________________________//

    // 4. Remove outside body air
    //_______________________________________________________________________//

    unsigned int *bounds_sub = new unsigned int[4 * s_volume_s[2]];

    for (unsigned int i = 0; i < 4 * volume_s[2]; i += subsample_f[2])
        bounds_sub[i] = bounds[i] / subsample_f[i < 2 * volume_s[2]];

    remove_outside_body_air(air_mask_sub, s_volume_s, bounds_sub);
    delete[] bounds;
    delete[] bounds_sub;

    //_______________________________________________________________________//

    // 5. Labeling in body air
    //_______________________________________________________________________//
    unsigned int *labeled = new unsigned int[s_volume_l];
    unsigned int *object_sizes = new unsigned int[s_volume_l];
    unsigned int label = 0;

    labeling_3d(const_cast<const bool *>(air_mask_sub), labeled, s_volume_s,
                object_sizes, label);
    delete[] air_mask_sub;
    //_______________________________________________________________________//

    // 6. Extracting lung candidates from labeled data according to the
    //    size and/or position
    //_______________________________________________________________________//

    bool *candidates = new bool[s_volume_l];
    float s_lung_volume_threshold = lung_volume_threshold
            / (subsample_f[0] * subsample_f[1] * subsample_f[2]);
    extract_lung_candidates(const_cast<const unsigned int *>(labeled),
                            const_cast<const unsigned int *>(s_volume_s),
                            object_sizes, label, candidates,
                            s_lung_volume_threshold, lung_assumed_center_n);

    delete[] labeled;
    delete[] object_sizes;
    //_______________________________________________________________________//

    // 7. Up-sample detected mask corresponding to the lungs
    //_______________________________________________________________________//

    bool *candidates_gpu;
    cudaMalloc((void **) &candidates_gpu, s_volume_l * sizeof(bool));
    cudaMemcpy(candidates_gpu, candidates, s_volume_l * sizeof(bool),
               cudaMemcpyHostToDevice);
    delete[] candidates;
    upsample_gpu<bool> <<<grid_sub, s_volume_s[0]>>>(air_mask_gpu,
            candidates_gpu,
            subsample_f[0],
            subsample_f[1],
            subsample_f[2]);

    cudaMemcpy(air_mask, air_mask_gpu, volume_l * sizeof(bool),
               cudaMemcpyDeviceToHost);
    cudaFree(air_mask_gpu);
    cudaFree(candidates_gpu);
    //_______________________________________________________________________//

    // 8. Re-label up-sampled data
    //_______________________________________________________________________//

    labeled = new unsigned int[volume_l];
    object_sizes = new unsigned int[volume_l];
    label = 0;
    labeling_3d(const_cast<const bool *>(air_mask), labeled, volume_s,
                object_sizes, label);

    candidates = new bool[volume_l];
    extract_lung_candidates(const_cast<const unsigned int *>(labeled), volume_s,
                            object_sizes, label, lungs_mask,
                            lung_volume_threshold, lung_assumed_center_n);

    delete[] object_sizes;
    delete[] labeled;
    delete[] air_mask;
    //_______________________________________________________________________//
}
