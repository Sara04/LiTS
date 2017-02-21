#include "preprocessor_cuda.h"


__global__ void gpu_normalization(float *cuda_volume, float *cuda_volume_n,
		                          bool change_direction,
		                          float lower_threshold_, float upper_threshold_,
		                          float minimum_value_, float maximum_value_)
{
	unsigned int in_idx = blockIdx.y * gridDim.x * blockDim.x +
			              blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int out_idx;

	if(change_direction)
		out_idx = blockIdx.y * gridDim.x * blockDim.x +
                  (gridDim.x - blockIdx.x - 1) * blockDim.x + threadIdx.x;
	else
		out_idx = in_idx;

	if(cuda_volume[in_idx] < lower_threshold_)
		cuda_volume_n[out_idx] = minimum_value_;
	else if(cuda_volume[in_idx] > upper_threshold_)
		cuda_volume_n[out_idx] = maximum_value_;
	else
		cuda_volume_n[out_idx] =
				(cuda_volume[in_idx] - lower_threshold_) *
				(maximum_value_ - minimum_value_) /
				(upper_threshold_ - lower_threshold_) + minimum_value_;
}

void preprocess_cuda(float *input_volume,
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
	cudaMemcpy(cuda_volume, input_volume, volume_bytes, cudaMemcpyHostToDevice);

	dim3 grid(h, d);

	gpu_normalization<<<grid, w>>>(cuda_volume, cuda_volume_n,
			                       change_direction,
			                       lower_threshold, upper_threshold,
			                       minimum_value, maximum_value);

	cudaMemcpy(input_volume, cuda_volume_n, volume_bytes, cudaMemcpyDeviceToHost);
	cudaFree(cuda_volume);
	cudaFree(cuda_volume_n);
}

