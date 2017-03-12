/*
 * liver_tumor_segment_cuda.cuh
 *
 *  Created on: Mar 12, 2017
 *      Author: sara
 */

#ifndef LIVER_TUMOR_SEGMENT_CUDA_CUH_
#define LIVER_TUMOR_SEGMENT_CUDA_CUH_

#include<list>
#include<iostream>


void develop(double *weights, double *biases,
             float *volumes, unsigned char *ground_truths,
             unsigned long *acc, unsigned int *lenghts,
             unsigned int *sizes, float *voxel_sizes,
             unsigned int n);

#endif /* LIVER_TUMOR_SEGMENT_CUDA_CUH_ */
