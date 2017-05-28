/*
 * preprocessor_cuda.cuh
 *
 *  Created on: Feb 14, 2017
 *      Author: sara
 */

#ifndef PRE_AND_POST_PROCESSOR_CUDA_CUH_
#define PRE_AND_POST_PROCESSOR_CUDA_CUH_

void reorient_permute(bool &reorient, bool &permute,
                      unsigned *cord, short *cornt,
                      unsigned *dord, short *dornt);

void preprocess_volume_cuda(float *volume_cpu,
                            unsigned w, unsigned h, unsigned d,
                            unsigned *direction, short *orientation,
                            float lower_threshold, float upper_threshold,
                            float minimum_value, float maximum_value);

void normalize_volume_cuda(float *volume_cpu,
                           unsigned w, unsigned h, unsigned d,
                           float lower_threshold, float upper_threshold,
                           float minimum_value, float maximum_value);

void filter_with_median_cuda(float *volume_cpu,
                             unsigned w, unsigned h, unsigned d,
                             int k);

void reorient_permute(bool &reorient, bool &permute,
                      unsigned *cord, short *cornt,
                      unsigned *dord, short *dornt);

void reorient_volume_cuda(float *volume_cpu,
                          unsigned w, unsigned h, unsigned d,
                          unsigned *cord, short *cornt,
                          unsigned *dord, short *dornt);

void reorient_segment_cuda(unsigned char *segmentation_cpu,
                           unsigned w, unsigned h, unsigned d,
                           unsigned *cord, short *cornt,
                           unsigned *dord, short *dornt);

#endif /* PRE_AND_POST_PROCESSOR_CUDA_CUH_ */
