#include "pre_and_post_processor_cuda.cuh"
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


template<typename T>
__global__ void gpu_reorient(T *data, T *data_o,
                             unsigned w, unsigned h, unsigned d,
                             unsigned ord0, unsigned ord1, unsigned ord2,
                             unsigned dord0, unsigned dord1, unsigned dord2,
                             short orient0, short orient1, short orient2,
                             short dorient0, short dorient1, short dorient2)
{

    unsigned int t = (ord0 == 0) * w + (ord1 == 0) * h + (ord2 == 0) * d;

    if(threadIdx.x < t)
    {
        unsigned int in_idx = blockIdx.y * gridDim.x * t +
                              blockIdx.x * t + threadIdx.x;
        unsigned int out_idx = 0;

        unsigned int crs[3] = {threadIdx.x, blockIdx.x, blockIdx.y};
        unsigned int dord[3] = {dord0, dord1, dord2};
        unsigned int s[3] = {w, h, d};
        unsigned int ord[3] = {ord0, ord1, ord2};

        out_idx += crs[ord[dord[0]]];

        if(orient1 == dorient1)
            out_idx += crs[ord[dord[1]]] * s[dord[0]];
        else
            out_idx += (s[dord[1]] - 1 - crs[ord[dord[1]]]) * s[dord[0]];

        if(orient2 == dorient2)
            out_idx += crs[dord[ord[2]]] * s[dord[0]] * s[dord[1]];
        else
            out_idx += (s[dord[2]] - 1 - crs[dord[ord[2]]]) * s[dord[0]] * s[dord[1]];

        data_o[out_idx] = data[in_idx];
    }
}

__global__ void gpu_normalize(float *cuda_volume,
                              unsigned w, unsigned h, unsigned d,
                              float lower_th_, float upper_th_,
                              float minimum_value_, float maximum_value_)
{

    unsigned int idx = blockIdx.y * gridDim.x * blockDim.x +
                       blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < w * h * d)
    {
        if (cuda_volume[idx] < lower_th_)
            cuda_volume[idx] = minimum_value_;
        else if (cuda_volume[idx] > upper_th_)
            cuda_volume[idx] = maximum_value_;
        else
            cuda_volume[idx] = (cuda_volume[idx] - lower_th_) *
                               (maximum_value_ - minimum_value_) /
                               (upper_th_ - lower_th_) + minimum_value_;
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
void preprocess_volume_cuda(float *input_volume,
                            unsigned int w, unsigned int h, unsigned int d,
                            unsigned int *order, short *orient,
                            float lower_threshold, float upper_threshold,
                            float minimum_value, float maximum_value)
{
    short dorient[3] = {0, 1, 1};
    unsigned dord[3] = {0, 1, 2};
    bool reorient = false;
    bool permute = false;
    for(unsigned int i = 1; i < 3; i++)
    {
        if(orient[i] != dorient[i])
            reorient = true;
        if(order[i] != i)
            permute = true;
    }
    float *cuda_volume;
    unsigned int volume_bytes = h * w * d * sizeof(float);

    cudaMalloc((void **) &cuda_volume, volume_bytes);
    cudaMemcpy(cuda_volume, input_volume, volume_bytes,
               cudaMemcpyHostToDevice);

    unsigned int i1, i2;
    i1 = (order[1] == 0) * w + (order[1] == 1) * h + (order[1] == 2) * d;
    i2 = (order[2] == 0) * w + (order[2] == 1) * h + (order[2] == 2) * d;

    dim3 grid(i1, i2);

    gpu_normalize<<<grid, MAX_THREADS>>>(cuda_volume,
                                         w, h, d,
                                         lower_threshold,
                                         upper_threshold,
                                         minimum_value,
                                         maximum_value);
    if(reorient or permute)
    {
        float *cuda_volume_o;
        cudaMalloc((void **) &cuda_volume_o, volume_bytes);
        gpu_reorient<float><<<grid, MAX_THREADS>>>(cuda_volume,
                                                   cuda_volume_o,
                                                   w, h, d,
                                                   order[0],
                                                   order[1],
                                                   order[2],
                                                   dord[0],
                                                   dord[1],
                                                   dord[2],
                                                   orient[0],
                                                   orient[1],
                                                   orient[2],
                                                   dorient[0],
                                                   dorient[1],
                                                   dorient[2]);

        cudaMemcpy(input_volume, cuda_volume_o, volume_bytes,
                   cudaMemcpyDeviceToHost);
        cudaFree(cuda_volume_o);
    }
    else
        cudaMemcpy(input_volume, cuda_volume, volume_bytes,
                   cudaMemcpyDeviceToHost);
    cudaFree(cuda_volume);
}


void normalize_volume_cuda(float *input_volume,
                           unsigned int w, unsigned int h, unsigned int d,
                           float lower_threshold, float upper_threshold,
                           float minimum_value, float maximum_value)
{

    float *cuda_volume;
    unsigned int volume_bytes = h * w * d * sizeof(float);

    cudaMalloc((void **) &cuda_volume, volume_bytes);
    cudaMemcpy(cuda_volume, input_volume, volume_bytes,
               cudaMemcpyHostToDevice);

    dim3 grid(h, d);

    gpu_normalize<<<grid, MAX_THREADS>>>(cuda_volume,
                                         w, h, d,
                                         lower_threshold,
                                         upper_threshold,
                                         minimum_value,
                                         maximum_value);

    cudaMemcpy(input_volume, cuda_volume, volume_bytes,
               cudaMemcpyDeviceToHost);
    cudaFree(cuda_volume);
}


void reorient_volume_cuda(float *input_volume,
                          unsigned int w, unsigned int h, unsigned int d,
                          unsigned *cord, short *corient,
                          unsigned *dord, short *dorient)
{
    bool reorient = false;
    bool permute = false;
    for(unsigned int i = 1; i < 3; i++)
    {
        if(corient[i] != dorient[i])
            reorient = true;
        if(cord[i] != dord[i])
            permute = true;
    }

    if(reorient or permute)
    {

        float *cuda_volume;
        float *cuda_volume_o;
        unsigned int volume_bytes = h * w * d * sizeof(float);

        cudaMalloc((void **) &cuda_volume, volume_bytes);
        cudaMemcpy(cuda_volume, input_volume, volume_bytes,
                   cudaMemcpyHostToDevice);
        cudaMalloc((void **) &cuda_volume_o, volume_bytes);

        unsigned int i1, i2;
        i1 = (cord[1] == 0) * w + (cord[1] == 1) * h + (cord[1] == 2) * d;
        i2 = (cord[2] == 0) * w + (cord[2] == 1) * h + (cord[2] == 2) * d;

        dim3 grid(i1, i2);

        gpu_reorient<float><<<grid, MAX_THREADS>>>(cuda_volume,
                                                   cuda_volume_o,
                                                   w, h, d,
                                                   cord[0],
                                                   cord[1],
                                                   cord[2],
                                                   dord[0],
                                                   dord[1],
                                                   dord[2],
                                                   corient[0],
                                                   corient[1],
                                                   corient[2],
                                                   dorient[0],
                                                   dorient[1],
                                                   dorient[2]);

        cudaMemcpy(input_volume, cuda_volume_o, volume_bytes,
                   cudaMemcpyDeviceToHost);
        cudaFree(cuda_volume_o);
        cudaFree(cuda_volume);
    }
}


void reorient_segmentation_cuda(unsigned char *input_segmentation,
                                unsigned int w, unsigned int h, unsigned int d,
                                unsigned *cord, short *corient,
                                unsigned *dord, short *dorient)
{
    bool reorient = false;
    bool permute = false;
    for(unsigned int i = 1; i < 3; i++)
    {
        if(corient[i] != dorient[i])
            reorient = true;
        if(cord[i] != dord[i])
            permute = true;
    }

    if(reorient or permute)
    {

        unsigned char *cuda_segmentation;
        unsigned char *cuda_segmentation_o;
        unsigned int segmentation_bytes = h * w * d * sizeof(unsigned char);

        cudaMalloc((void **) &cuda_segmentation, segmentation_bytes);
        cudaMemcpy(cuda_segmentation, input_segmentation, segmentation_bytes,
                   cudaMemcpyHostToDevice);
        cudaMalloc((void **) &cuda_segmentation_o, segmentation_bytes);

        unsigned int i1, i2;
        i1 = (cord[1] == 0) * w + (cord[1] == 1) * h + (cord[1] == 2) * d;
        i2 = (cord[2] == 0) * w + (cord[2] == 1) * h + (cord[2] == 2) * d;

        dim3 grid(i1, i2);
        gpu_reorient<unsigned char><<<grid, MAX_THREADS>>>(cuda_segmentation,
                                                           cuda_segmentation_o,
                                                           w, h, d,
                                                           cord[0],
                                                           cord[1],
                                                           cord[2],
                                                           dord[0],
                                                           dord[1],
                                                           dord[2],
                                                           corient[0],
                                                           corient[1],
                                                           corient[2],
                                                           dorient[0],
                                                           dorient[1],
                                                           dorient[2]);

        cudaMemcpy(input_segmentation, cuda_segmentation_o, segmentation_bytes,
                   cudaMemcpyDeviceToHost);
        cudaFree(cuda_segmentation_o);
        cudaFree(cuda_segmentation);
    }
}
