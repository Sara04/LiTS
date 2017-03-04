#include "preprocessor_cuda.h"


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
 * 		change_direction - flag indicating weather to change volume and
 * 			segmentation orientation along front-back body axis
 *
 * 		lower_th_: lower limit for voxel intensity
 * 		upper_th_: upper limit for voxel intensity
 * 		minimum_value_: minimum voxel intensity value in the
 * 			normalized voxel range
 * 		maximum_value_: maximum voxel intensity value in the
 * 			normalized voxel range
 *
 * 		????????????????????????????????????????????????????????????????
 * 		Should be adapted to accept several volumes and/or segmentations
 * 		at one time
 * 		????????????????????????????????????????????????????????????????
 */
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
 * 		change_direction - flag indicating weather to change volume and
 * 			segmentation orientation along front-back body axis
 *
 * 		lower_threshold: lower limit for voxel intensity
 * 		upper_threshold: upper limit for voxel intensity
 * 		minimum_value: minimum voxel intensity value in the
 * 			normalized voxel range
 * 		maximum_value: maximum voxel intensity value in the
 * 			normalized voxel range
 * 		approach: selection between "itk" (cpu) and "cuda" (gpu)
 * 			normalization
 *
 *
 * 		????????????????????????????????????????????????????????????????
 * 		Should be adapted to accept only volume
 * 		Should be adapted to accept several volumes and/or segmentations
 * 		at one time
 * 		Should be changed in order to use all threads in block instead
 * 		of the number corresponding volume/segmentation width
 * 		????????????????????????????????????????????????????????????????
 *
 */
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

