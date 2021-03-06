/*
 * segmentator_cuda_kernels.cuh
 *
 *  Created on: Apr 8, 2017
 *      Author: sara
 */

#ifndef SEGMENTATOR_CUDA_KERNELS_CUH_
#define SEGMENTATOR_CUDA_KERNELS_CUH_

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
__global__ inline void volume_air_segmentation_gpu(const float *volume,
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
__global__ inline void detect_body_bounds_gpu(const bool *air_mask, unsigned *bounds,
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

/******************************************************************************
 * get_organ_mask_accs_gpu: accumulate organ mask values over each of the axes
 *
 * Arguments:
 *      meta_mask: pointer to the array containing meta segmentation
 *      value: value of the organ's label
 *      accs: accumulators
******************************************************************************/
__global__ inline void get_organ_mask_accs_gpu(const unsigned char *meta_mask,
                                               unsigned char value,
                                               unsigned int *accs)
{
    unsigned int idx = blockIdx.y * gridDim.x * blockDim.x
                       + blockIdx.x * blockDim.x + threadIdx.x;

    bool i = (meta_mask[idx] == value);
    atomicAdd(&accs[threadIdx.x], i);
    atomicAdd(&accs[blockDim.x + blockIdx.x], i);
    atomicAdd(&accs[gridDim.x + blockDim.x + blockIdx.y], i);
}

/******************************************************************************
 * get_organ_mask_accs_gpu: accumulate organ mask values over each of the axes
 *
 * Arguments:
 *      meta_mask: pointer to the array containing meta segmentation
 *      value: value of the organ's label
 *      accs: accumulators
******************************************************************************/
__global__ inline void get_organ_mask_accs_gpu_multiple(
        const unsigned char *meta_mask, unsigned *lenghts, unsigned *S,
        unsigned char value, unsigned int *accs, unsigned n_samples)
{
    unsigned int idx = blockIdx.y * gridDim.x * blockDim.x
                       + blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < lenghts[n_samples])
    {
        bool v = (meta_mask[idx] == value);
        if(v)
        {
            unsigned int shift = 0;
            unsigned int j = 0;
            for(unsigned int i = 0; i < n_samples; i++)
            {
                j = i;
                if (i > 0)
                    shift += (S[3 * (i - 1)] +
                              S[3 * (i - 1) + 1] +
                              S[3 * (i - 1) + 2]);
                if(idx < lenghts[i + 1])
                    break;
            }

            unsigned p = (idx - lenghts[j]);
            unsigned s = p / (S[3 * j] * S[3 * j + 1]);
            unsigned r = (p - s * S[3 * j] * S[3 * j + 1]) / S[3 * j];
            unsigned c = p - s * S[3 * j] * S[3 * j + 1] - r * S[3 * j];

            atomicAdd(&accs[shift + c], 1);
            atomicAdd(&accs[shift + S[3 * j] + r], 1);
            atomicAdd(&accs[shift + S[3 * j] + S[3 * j + 1] + s], 1);
        }
    }
}


__global__ inline void resize_volume_slice_and_crop(float *images,
		                                            float *images_rs,
                                                    unsigned *bounds,
                                                    unsigned *lenghts,
                                                    float *rotations,
                                                    unsigned N_sl,
                                                    unsigned N_aug,
                                                    unsigned w_rs,
                                                    unsigned h_rs)
{

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < (N_sl * N_aug * w_rs * h_rs))
    {
        unsigned s_idx = 0;
        for(unsigned int i = 0; i < N_sl; i++)
        {
            s_idx = i;
            if(idx < (i + 1) * N_aug * w_rs * h_rs)
                break;
        }
        unsigned b_idx = s_idx * N_aug;
        for(unsigned int i = s_idx * N_aug; i < (s_idx + 1) * N_aug; i++)
        {
            b_idx = i;
            if(idx < (i + 1) * w_rs * h_rs)
                break;
        }

        unsigned p_rs = (idx - s_idx * w_rs * h_rs * N_aug);
        unsigned a_rs = p_rs / ( w_rs * h_rs);
        unsigned r_rs = (p_rs - a_rs * w_rs * h_rs) / w_rs;
        unsigned c_rs = p_rs - a_rs * w_rs * h_rs - r_rs * w_rs;

        float w_sc = float(bounds[4 * (s_idx * N_aug + a_rs) + 1] + 1 -
        		           bounds[4 * (s_idx * N_aug + a_rs)]) / w_rs;
        float h_sc = float(bounds[4 * (s_idx * N_aug + a_rs) + 3] + 1 -
        		           bounds[4 * (s_idx * N_aug + a_rs) + 2]) / h_rs;

        float c_new_rs =\
        		rotations[4 * b_idx] * float(c_rs - float(w_rs) / 2) +
        		float(w_rs) / 2 +
        		rotations[4 * b_idx + 1] * float(r_rs - float(h_rs) / 2);

        float r_new_rs =\
        		rotations[4 * b_idx + 2] * float(c_rs - float(w_rs) / 2) +
        		rotations[4 * b_idx + 3] * float(r_rs - float(h_rs) / 2) +
        		float(h_rs) / 2;

        float dw = c_new_rs * w_sc - floor(c_new_rs * w_sc);
        float dh = r_new_rs * h_sc - floor(r_new_rs * h_sc);
        unsigned c = bounds[4 * b_idx] + floor(c_new_rs * w_sc);
        unsigned r = bounds[4 * b_idx + 2] + floor(r_new_rs * h_sc);

        images_rs[idx] =\
        		images[lenghts[s_idx] + 512*(r + 1) + (c + 1)] * dw * dh +
                images[lenghts[s_idx] + 512*r + c] * (1.0 - dw) * (1.0 - dh) +
        	    images[lenghts[s_idx] + 512*(r + 1) + c] * dw * (1.0 - dh) +
        	    images[lenghts[s_idx] + 512*r + c + 1] * (1.0 - dw) * dh;
    }
}
/*
resize_slice_and_crop<<<no_blocks, MAX_THREADS>>>
		(sls_V_d, sls_V_rs_d, sls_bx_d, Ls_bx_d,
	     N_sl * N_aug, w_rs, h_rs);
__global__ inline void resize_slice_and_crop(float *images, float *images_rs,
                                             unsigned int *bounds,
                                             unsigned int *lenghts,
                                             unsigned int N_sl,
                                             unsigned int N_aug,
                                             unsigned int w_rs,
                                             unsigned int h_rs)
{

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < (N_sl * N_aug * w_rs * h_rs))
    {
        unsigned s_idx = 0;
        for(unsigned int i = 0; i < N_sl; i++)
        {
            s_idx = i;
            if(idx < (i + 1) * w_rs * h_rs * N_aug)
                break;
        }

        unsigned p_rs = (idx - s_idx * w_rs * h_rs * N_aug);
        unsigned a_rs = p_rs / ( w_rs * h_rs);
        unsigned r_rs = (p_rs - a_rs * w_rs * h_rs) / w_rs;
        unsigned c_rs = p_rs - a_rs * w_rs * h_rs - r_rs * w_rs;

        float w_sc = float(bounds[4 * (s_idx * N_aug + a_rs) + 1] + 1 - bounds[4 * (s_idx * N_aug + a_rs)]) / w_rs;
        float h_sc = float(bounds[4 * (s_idx * N_aug + a_rs) + 3] + 1 - bounds[4 * (s_idx * N_aug + a_rs) + 2]) / h_rs;

        float dw = float(c_rs) * w_sc - floor(float(c_rs) * w_sc);
        float dh = float(r_rs) * h_sc - floor(float(r_rs) * h_sc);
        unsigned c = bounds[4 * s_idx] + floor(float(c_rs) * w_sc);
        unsigned r = bounds[4 * s_idx + 2] + floor(float(r_rs) * h_sc);

        images_rs[idx] = images[lenghts[s_idx] + 512*(r + 1) + (c + 1)] * (dw * dh) +
                         images[lenghts[s_idx] + 512*r + c] * ((1.0 - dw) * (1.0 - dh)) +
                         images[lenghts[s_idx] + 512*(r + 1) + c] * (dw * (1.0 - dh)) +
                         images[lenghts[s_idx] + 512*r + c + 1] * ((1.0 - dw) * dh);

    }

}
*/
__global__ inline void resize_gt_slice_and_crop(unsigned char *images,
		                                        unsigned char *images_rs,
                                                unsigned *bounds,
                                                unsigned *lenghts,
                                                float *rotations,
                                                unsigned N_sl,
                                                unsigned N_aug,
                                                unsigned w_rs,
                                                unsigned h_rs)
{

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < (N_sl * N_aug * w_rs * h_rs))
    {
        unsigned s_idx = 0;
        for(unsigned int i = 0; i < N_sl; i++)
        {
            s_idx = i;
            if(idx < (i + 1) * N_aug * w_rs * h_rs)
                break;
        }
        unsigned b_idx = s_idx * N_aug;
        for(unsigned int i = s_idx * N_aug; i < (s_idx + 1) * N_aug; i++)
        {
            b_idx = i;
            if(idx < (i + 1) * w_rs * h_rs)
                break;
        }

        unsigned p_rs = (idx - s_idx * w_rs * h_rs * N_aug);
        unsigned a_rs = p_rs / ( w_rs * h_rs);
        unsigned r_rs = (p_rs - a_rs * w_rs * h_rs) / w_rs;
        unsigned c_rs = p_rs - a_rs * w_rs * h_rs - r_rs * w_rs;

        float w_sc = float(bounds[4 * (s_idx * N_aug + a_rs) + 1] + 1 - bounds[4 * (s_idx * N_aug + a_rs)]) / w_rs;
        float h_sc = float(bounds[4 * (s_idx * N_aug + a_rs) + 3] + 1 - bounds[4 * (s_idx * N_aug + a_rs) + 2]) / h_rs;

        float c_new_rs = rotations[4 * b_idx] * float(c_rs - float(w_rs) / 2) + float(w_rs) / 2 +
        		         rotations[4 * b_idx + 1] * float(r_rs - float(h_rs) / 2);

        float r_new_rs = rotations[4 * b_idx + 2] * float(c_rs - float(w_rs) / 2) +
        		         rotations[4 * b_idx + 3] * float(r_rs - float(h_rs) / 2) + float(h_rs) / 2;

        float dw = c_new_rs * w_sc - floor(c_new_rs * w_sc);
        float dh = r_new_rs * h_sc - floor(r_new_rs * h_sc);
        unsigned c = bounds[4 * b_idx] + floor(c_new_rs * w_sc);
        unsigned r = bounds[4 * b_idx + 2] + floor(r_new_rs * h_sc);

        float tmp;
        tmp = float(images[lenghts[s_idx] + 512*(r + 1) + (c + 1)]) * dw * dh +
              float(images[lenghts[s_idx] + 512*r + c]) * (1.0 - dw) * (1.0 - dh) +
        	  float(images[lenghts[s_idx] + 512*(r + 1) + c]) * dw * (1.0 - dh) +
        	  float(images[lenghts[s_idx] + 512*r + c + 1]) * (1.0 - dw) * dh;

        images_rs[idx] = (unsigned char)(tmp + 0.5);
    }
}

template<typename T>
__global__ inline void flip_slices(T *images_rs, unsigned int N_sl,
                                   unsigned w_rs, unsigned h_rs)
{

    unsigned int idx = blockIdx.y * gridDim.x * blockDim.x +
                       blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < (N_sl * w_rs * h_rs))
    {
        unsigned s_idx = 0;
        for(unsigned int i = 0; i < N_sl; i++)
        {
            s_idx = i;
            if(idx < (i + 1) * w_rs * h_rs)
                break;
        }

        unsigned p_rs = (idx - s_idx * w_rs * h_rs);
        unsigned r_rs = p_rs / w_rs;
        unsigned c_rs = p_rs - r_rs * w_rs;


        images_rs[N_sl * w_rs * h_rs +
                  s_idx * w_rs * h_rs +
                  r_rs * w_rs + (w_rs - 1 - c_rs)] = images_rs[idx];

    }

}

__global__ inline void accumulation_gpu(float * input_data,
                                 float * accumulator,
                                 unsigned in_len,
                                 unsigned acc_len)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < in_len)
    {
        unsigned idx_acc = idx % acc_len;

        atomicAdd(&accumulator[idx_acc], input_data[idx]);
    }
}

__global__ inline void squared_accumulation_gpu(float * input_data,
                                                 float * accumulator,
                                                 float *mean,
                                                 unsigned in_len,
                                                 unsigned acc_len)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < in_len)
    {
        unsigned idx_acc = idx % acc_len;

        atomicAdd(&accumulator[idx_acc],
                  pow(input_data[idx] - mean[idx_acc], 2));
    }
}

__global__ inline void mean_std_normalization(float *data,
                                              float *mean,
                                              float *std,
                                              unsigned data_len,
                                              unsigned feat_len)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < data_len)
    {
        unsigned ms_idx = idx % feat_len;
        data[idx] -= mean[ms_idx];
        data[idx] /= std[ms_idx];
    }
}
#endif /* SEGMENTATOR_CUDA_KERNELS_CUH_ */
