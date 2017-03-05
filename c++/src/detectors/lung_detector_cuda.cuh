/*
 * lung_detector_cuda.cuh
 *
 *  Created on: Mar 4, 2017
 *      Author: sara
 */

#ifndef LUNG_DETECTOR_CUDA_CUH_
#define LUNG_DETECTOR_CUDA_CUH_

#include "../tools/tools.h"

/*
 * Functions used for lung detection and segmentation
 *
 * Functions:
 * 		is_in_body_box: verify if pixel is within
 * 			slice bounds
 * 		remove_outside_body_air: remove binary objects
 * 			that belong to the outside body air from
 * 			air binary mask
 * 		extract_lung_labels: extract objects with given
 * 			labels from labeled objects
 * 		extract_lung_candidates: extract objects that
 * 		    according to the size and/or position
 * 			most likely belong to lungs
 * 		segment_lungs: segmentation of lungs
 */

bool is_in_body_box(const unsigned int *slice_bounds,
		            unsigned int c_idx, unsigned int r_idx);

void remove_outside_body_air(bool *air_mask,
		                     const unsigned int *size,
		                     const unsigned int *bounds);

void extract_lung_labels(const unsigned int *labeled,
		                 bool *candidates,
		                 const unsigned int *size,
		                 const unsigned int *main_labels,
		                 unsigned int count);

void extract_lung_candidates(const unsigned int *labeled,
		                     const unsigned int *size,
		                     unsigned int *object_sizes,
		                     unsigned int &label,
		                     bool *candidates,
		                     float &size_threshold);

void segment_lungs(const float *volume,
		           const unsigned int *volume_size,
			       const unsigned int *subsample_factor,
			       const float *lung_assumed_center_n,
			       const unsigned int *body_bounds_th,
			       float lung_volume_threshold,
			       float air_threshold);


#endif /* LUNG_DETECTOR_CUDA_CUH_ */
