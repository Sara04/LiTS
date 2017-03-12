/*
 * liver_detector_cuda.cuh
 *
 *  Created on: Mar 5, 2017
 *      Author: sara
 */

#ifndef LIVER_DETECTOR_CUDA_CUH_
#define LIVER_DETECTOR_CUDA_CUH_


void estimate_liver_lung_size(const bool *lungs_mask,
                              const unsigned char *liver_mask,
                              const unsigned int *size,
                              const float *voxel_size,
                              const unsigned int *body_bounds,
                              unsigned int *liver_bounds);



#endif /* LIVER_DETECTOR_CUDA_CUH_ */
