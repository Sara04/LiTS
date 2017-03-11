#include "preprocessor_cuda.cuh"
#define MAX_THREADS 1024

#include <iostream>

/*
 * cuda kernel
 * gpu_normalization: normalize voxel intensities and flip volume and
 * 		segmentation if necessary
 *
 * Arguments:
 * 		cuda_volume: pointer to the array containing original
 * 			volume
 * 		cuda_volume_n: pointer to the array where normalized (and
 * 			re-oriented, if necessary) volume would be placed
 * 		cuda_segmentation: pointer to the array containing original
 * 			segmentation
 * 		cuda_segmentation_o: pointer to the array where re-oriented,
 * 			if necessary, segmentation would be placed
 * 		segmentation_flag: flag indicating if segmentation exists
 * 		    and if so if its axes should be ordered or flipped
 *      w - volume/segmentation width
 *      h - volume/segmentation height
 *      d - volume/segmentation number of slices
 *      order0, order1, order2 - order of axes
 *      orient0, orient1, orient2 - orientation of ordered axes
 * 		lower_th_: lower limit for voxel intensity
 * 		upper_th_: upper limit for voxel intensity
 * 		minimum_value_: minimum voxel intensity value in the
 * 			normalized voxel range
 * 		maximum_value_: maximum voxel intensity value in the
 * 			normalized voxel range
 *
 */
__global__ void gpu_normalization(float *cuda_volume, float *cuda_volume_n,
                                  unsigned char *cuda_segmentation,
                                  unsigned char *cuda_segmentation_o,
                                  bool segmentation_flag,
                                  unsigned w, unsigned h, unsigned d,
                                  unsigned ord0, unsigned ord1, unsigned ord2,
                                  short orient0, short orient1, short orient2,
                                  float lower_th_, float upper_th_,
                                  float minimum_value_, float maximum_value_)
{
    unsigned int t = (ord0 == 0) * w + (ord1 == 0) * h + (ord2 == 0) * d;

    if(threadIdx.x < t)
    {
        unsigned int in_idx = blockIdx.y * gridDim.x * t +
                              blockIdx.x * t + threadIdx.x;
        unsigned int out_idx = 0;

        unsigned int crs[3] = {threadIdx.x, blockIdx.x, blockIdx.y};

        out_idx += crs[ord0];

        if(orient1 == 1)
            out_idx += crs[ord1] * w;
        else
            out_idx += (h - 1 - crs[ord1]) * w;

        if(orient2 == 1)
            out_idx += crs[ord2] * h * w;
        else
            out_idx += (d - 1 - crs[ord2]) * w * h;

        if (cuda_volume[in_idx] < lower_th_)
            cuda_volume_n[out_idx] = minimum_value_;
        else if (cuda_volume[in_idx] > upper_th_)
            cuda_volume_n[out_idx] = maximum_value_;
        else
            cuda_volume_n[out_idx] = (cuda_volume[in_idx] - lower_th_) *
                                     (maximum_value_ - minimum_value_) /
                                     (upper_th_ - lower_th_) + minimum_value_;

        if(segmentation_flag)
            cuda_segmentation_o[out_idx] = cuda_segmentation[in_idx];

    }
}

/*
 * preprocess_cuda: normalize voxel intensities and flip volume and
 * 		segmentation if necessary
 *
 * Arguments:
 * 		volume_cpu: pointer to the array containing volume
 * 		segmentation_cpu: pointer to the array containing segmentation
 * 		h - volume/segmentation height
 * 		w - volume/segmentation width
 * 		d - volume/segmentation depth
 *      order - order of axes
 *      orient - orientation of axes
 * 		lower_threshold: lower limit for voxel intensity
 * 		upper_threshold: upper limit for voxel intensity
 * 		minimum_value: minimum voxel intensity value in the
 * 			normalized voxel range
 * 		maximum_value: maximum voxel intensity value in the
 * 			normalized voxel range
 * 		approach: selection between "itk" (cpu) and "cuda" (gpu)
 * 			normalization
 *
 */
void preprocess_cuda(float *input_volume, unsigned char *input_segmentation,
                     unsigned int h, unsigned int w, unsigned int d,
                     unsigned int *order, short *orient,
                     float lower_threshold, float upper_threshold,
                     float minimum_value, float maximum_value)
{
    short desired_orientation[3] = {0, 1, 1};
    bool reorient = false;
    bool permute = false;
    for(unsigned int i = 1; i < 3; i++)
    {
        if(orient[i] != desired_orientation[i])
            reorient = true;
        if(order[i] != i)
            permute = true;
    }
    float *cuda_volume;
    float *cuda_volume_n;
    unsigned int volume_bytes = h * w * d * sizeof(float);

    cudaMalloc((void **) &cuda_volume, volume_bytes);
    cudaMalloc((void **) &cuda_volume_n, volume_bytes);
    cudaMemcpy(cuda_volume, input_volume, volume_bytes,
               cudaMemcpyHostToDevice);
    unsigned char *cuda_segmentation;
    unsigned char *cuda_segmentation_o;
    unsigned int segment_bytes = sizeof(unsigned char);

    if ((reorient or permute) and input_segmentation)
    {
        segment_bytes *= (h * w * d);
        cudaMalloc((void **) &cuda_segmentation, segment_bytes);
        cudaMalloc((void **) &cuda_segmentation_o, segment_bytes);
        cudaMemcpy(cuda_segmentation, input_segmentation, segment_bytes,
                   cudaMemcpyHostToDevice);
    }

    unsigned int i1, i2;
    i1 = (order[1] == 0) * w + (order[1] == 1) * h + (order[1] == 2) * d;
    i2 = (order[2] == 0) * w + (order[2] == 1) * h + (order[2] == 2) * d;

    dim3 grid(i1, i2);

    gpu_normalization<<<grid, MAX_THREADS>>>(cuda_volume,
                                             cuda_volume_n,
                                             cuda_segmentation,
                                             cuda_segmentation_o,
                                             (reorient or permute) and
                                             input_segmentation,
                                             w, h, d,
                                             order[0], order[1], order[2],
                                             orient[0], orient[1], orient[2],
                                             lower_threshold, upper_threshold,
                                             minimum_value, maximum_value);

    cudaMemcpy(input_volume, cuda_volume_n, volume_bytes,
               cudaMemcpyDeviceToHost);
    cudaFree(cuda_volume);
    cudaFree(cuda_volume_n);

    if ((reorient or permute) and input_segmentation)
    {
        cudaMemcpy(input_segmentation, cuda_segmentation_o, segment_bytes,
                   cudaMemcpyDeviceToHost);
        cudaFree(cuda_segmentation);
        cudaFree(cuda_segmentation_o);
    }
}

