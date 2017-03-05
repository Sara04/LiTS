/*
 * preprocessor_cuda.cuh
 *
 *  Created on: Feb 14, 2017
 *      Author: sara
 */

#ifndef PREPROCESSOR_CUDA_CUH_
#define PREPROCESSOR_CUDA_CUH_

/*
 * preprocess_cuda: normalize voxel intensities and flip volume and
 * 		segmentation if necessary
 */
void preprocess_cuda(float *volume_cpu, unsigned char *segmentation_cpu,
                     unsigned int h, unsigned int w, unsigned int d,
                     bool change_direction, float lower_threshold,
                     float upper_threshold, float minimum_value,
                     float maximum_value);

#endif /* PREPROCESSOR_CUDA_CUH_ */
