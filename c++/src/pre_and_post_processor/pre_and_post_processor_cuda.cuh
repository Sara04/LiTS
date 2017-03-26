/*
 * preprocessor_cuda.cuh
 *
 *  Created on: Feb 14, 2017
 *      Author: sara
 */

#ifndef PRE_AND_POST_PROCESSOR_CUDA_CUH_
#define PRE_AND_POST_PROCESSOR_CUDA_CUH_

/*
 * preprocess_cuda: normalize voxel intensities and
 * order and/or flip volume and segmentation if necessary
 */
void preprocess_volume_cuda(float *volume_cpu,
                            unsigned int w, unsigned int h, unsigned int d,
                            unsigned int *direction, short *orientation,
                            float lower_threshold, float upper_threshold,
                            float minimum_value, float maximum_value);

void normalize_volume_cuda(float *volume_cpu,
                           unsigned int w, unsigned int h, unsigned int d,
                           float lower_threshold, float upper_threshold,
                           float minimum_value, float maximum_value);

void reorient_volume_cuda(float *volume_cpu,
                          unsigned int w, unsigned int h, unsigned int d,
                          unsigned *cord, short *corient,
                          unsigned *dord, short *dorient);

void reorient_segmentation_cuda(unsigned char *segmentation_cpu,
                                unsigned int w, unsigned int h, unsigned int d,
                                unsigned *cord, short *corient,
                                unsigned *dord, short *dorient);

#endif /* PRE_AND_POST_PROCESSOR_CUDA_CUH_ */
