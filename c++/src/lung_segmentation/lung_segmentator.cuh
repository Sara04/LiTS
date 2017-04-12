/*
 * lung_detector.cuh
 *
 *  Created on: Mar 4, 2017
 *      Author: sara
 */

#ifndef LUNG_SEGMENTATOR_CUDA_CUH_
#define LUNG_SEGMENTATOR_CUDA_CUH_

#include "../tools/tools.h"
#include "lung_segmentator_tools.h"

/******************************************************************************
 * Function used for lung detection and segmentation
 *
 * Functions:
 * 		segment_lungs: segmentation of lungs by following steps:
 * 		    - detecting air regions
 * 		    - detecting body bounds
 * 		    - down-sampling in order to decrease computational time
 * 		    - removing outside body air
 * 		    - determining central slice of the largest in-body air object
 * 		        (and removing air objects that are far from this)
 * 		    - labeling in-body air
 * 		    - extraction of the lungs candidates according to the size and/or
 * 		        position
 * 		    - up-sampling
 * 		    - lung mask re-labeling and re-fining
******************************************************************************/
void segment_lungs(const float *volume, const unsigned int *volume_size,
                   bool *lungs_mask, const unsigned int *subsample_factor,
                   float *lung_assumed_center_n,
                   const unsigned int *body_bounds_th,
                   float lung_volume_threshold, float air_threshold);

#endif /* LUNG_SEGMENTATOR_CUDA_CUH_ */
