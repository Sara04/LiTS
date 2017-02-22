#include "preprocessor_cuda.h"


__global__ void gpu_normalization(float *cuda_volume, float *cuda_volume_n,
		                          unsigned char *cuda_segmentation,
		                          unsigned char *cuda_segmentation_o,
		                          bool change_direction,
		                          float lower_th_, float upper_th_,
		                          float minimum_value_, float maximum_value_)
{
	unsigned int in_idx = blockIdx.y * gridDim.x * blockDim.x +
			              blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int out_idx;

	if(change_direction)
	{
		out_idx = blockIdx.y * gridDim.x * blockDim.x +
                  (gridDim.x - blockIdx.x - 1) * blockDim.x + threadIdx.x;
	    cuda_segmentation_o[out_idx] = cuda_segmentation[in_idx];
	}
	else
		out_idx = in_idx;

	if(cuda_volume[in_idx] < lower_th_)
		cuda_volume_n[out_idx] = minimum_value_;
	else if(cuda_volume[in_idx] > upper_th_)
		cuda_volume_n[out_idx] = maximum_value_;
	else
		cuda_volume_n[out_idx] =
				(cuda_volume[in_idx] - lower_th_) *
				(maximum_value_ - minimum_value_) /
				(upper_th_ - lower_th_) + minimum_value_;
}

void preprocess_cuda(float *input_volume, unsigned char *input_segmentation,
		             unsigned int h, unsigned int w, unsigned int d,
		             bool change_direction,
		             float lower_threshold, float upper_threshold,
		             float minimum_value, float maximum_value)
{

	float *cuda_volume;
	float *cuda_volume_n;
	unsigned int volume_bytes = h * w * d * sizeof(float);

	cudaMalloc((void **)&cuda_volume, volume_bytes);
	cudaMalloc((void **)&cuda_volume_n, volume_bytes);
	cudaMemcpy(cuda_volume, input_volume, volume_bytes,
			   cudaMemcpyHostToDevice);

	unsigned char *cuda_segmentation;
	unsigned char *cuda_segmentation_o;
	unsigned int segment_bytes = sizeof(unsigned char);
	if(change_direction)
	{
		segment_bytes *= (h * w * d);
		cudaMalloc((void **)&cuda_segmentation, segment_bytes);
		cudaMalloc((void **)&cuda_segmentation_o, segment_bytes);
		cudaMemcpy(cuda_segmentation, input_segmentation, segment_bytes,
				   cudaMemcpyHostToDevice);
	}

	dim3 grid(h, d);

	gpu_normalization<<<grid, w>>>(cuda_volume, cuda_volume_n,
			                       cuda_segmentation, cuda_segmentation_o,
			                       change_direction,
			                       lower_threshold, upper_threshold,
			                       minimum_value, maximum_value);

	cudaMemcpy(input_volume, cuda_volume_n, volume_bytes,
			   cudaMemcpyDeviceToHost);
	cudaFree(cuda_volume);
	cudaFree(cuda_volume_n);

	if(change_direction)
	{
		cudaMemcpy(input_segmentation, cuda_segmentation_o, segment_bytes,
				   cudaMemcpyDeviceToHost);
		cudaFree(cuda_segmentation);
		cudaFree(cuda_segmentation_o);
	}
}

