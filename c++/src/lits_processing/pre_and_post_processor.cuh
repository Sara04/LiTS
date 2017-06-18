/*
 * preprocessor_cuda.cuh
 *
 *  Created on: Feb 14, 2017
 *      Author: sara
 */

#ifndef PRE_AND_POST_PROCESSOR_CUDA_CUH_
#define PRE_AND_POST_PROCESSOR_CUDA_CUH_

/******************************************************************************
 * reorient_permute: determine if there is a need to re-orient and/or permute
 * axes
 *
 * Arguments:
 * 		reorient: flag whether to re-orient axis
 * 		permute: flag whether to permute axis
 * 		cord: current axis order
 * 		cornt: current axis orientations
 * 		dord: desired axis order
 * 		dornt: desired axis orientation
 *****************************************************************************/
void reorient_permute(bool &reorient, bool &permute,
                      unsigned *cord, short *cornt,
                      unsigned *dord, short *dornt);

/******************************************************************************
 * preprocess_volume_cuda: normalize voxel intensities and re-orient volume
 * axes if necessary
 *
 * Arguments:
 * 		volume_cpu: volume to be processed
 * 		w: volume width
 * 		h: volume height
 * 		d: volume depth / number of slices
 * 		cord: current order of axis
 * 		cornt: current orientation of axis
 *      lower_threshold: lower limit for voxel intensity
 *      upper_threshold: upper limit for voxel intensity
 *      minimum_value: minimum voxel intensity value in the
 *          normalized voxel range
 *      maximum_value: maximum voxel intensity value in the
 *          normalized voxel range
 *****************************************************************************/
void preprocess_volume_cuda(float *volume_cpu,
                            unsigned w, unsigned h, unsigned d,
                            unsigned *cord, short *cornt,
                            float lower_threshold, float upper_threshold,
                            float minimum_value, float maximum_value);

/******************************************************************************
 * normalize_volume_cuda: normalize voxel intensities
 *
 * Arguments:
 *      volume_cpu: volume to be processed
 *      w: volume width
 *      h: volume height
 *      d: volume depth / number of slices
 *      lower_threshold: lower limit for voxel intensity
 *      upper_threshold: upper limit for voxel intensity
 *      minimum_value: minimum voxel intensity value in the
 *          normalized voxel range
 *      maximum_value: maximum voxel intensity value in the
 *          normalized voxel range
 *****************************************************************************/
void normalize_volume_cuda(float *volume_cpu,
                           unsigned w, unsigned h, unsigned d,
                           float lower_threshold, float upper_threshold,
                           float minimum_value, float maximum_value);

/******************************************************************************
 * filter_with_median_cuda: de-noise volume with median filter
 *
 * Arguments:
 *      volume_cpu: volume to be processed
 *      w: volume width
 *      h: volume height
 *      d: volume depth / number of slices
 *      k: median kernel's size
 *****************************************************************************/
void filter_with_median_cuda(float *volume_cpu,
                             unsigned w, unsigned h, unsigned d, int k);

/******************************************************************************
 * reorient_volume_cuda: re-orient axes of volume if necessary
 *
 * Arguments:
 *      volume_cpu: volume to be reoriented
 *      w: volume width
 *      h: volume height
 *      d: volume depth / number of slices
 *      cord - current order of the axes
 *      cornt - current orientation of the axes
 *      dord - desired order of the axes
 *      dornt - desired orientation of the axes
 *****************************************************************************/
void reorient_volume_cuda(float *volume_cpu,
                          unsigned w, unsigned h, unsigned d,
                          unsigned *cord, short *cornt,
                          unsigned *dord, short *dornt);

/******************************************************************************
 * reorient_segmentation_cuda: re-orient axes of segmentation if necessary
 *
 * Arguments:
 *      segmentation_cpu: segmentation to be reoriented
 *      w: volume width
 *      h: volume height
 *      d: volume depth / number of slices
 *      cord - current order of the axes
 *      cornt - current orientation of the axes
 *      dord - desired order of the axes
 *      dornt - desired orientation of the axes
 *****************************************************************************/
void reorient_segment_cuda(unsigned char *segmentation_cpu,
                           unsigned w, unsigned h, unsigned d,
                           unsigned *cord, short *cornt,
                           unsigned *dord, short *dornt);

#endif /* PRE_AND_POST_PROCESSOR_CUDA_CUH_ */
