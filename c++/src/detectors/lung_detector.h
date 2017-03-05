/*
 * lung_detector.h
 *
 *  Created on: Feb 25, 2017
 *      Author: sara
 */

#ifndef LUNG_DETECTOR_H_
#define LUNG_DETECTOR_H_

#include "../lits_scan/lits_scan.h"
#include "lung_detector_cuda.cuh"

/*
 * LiTS_lung_detector class for detecting binary mask
 * that belongs to lungs. Segmentation of lungs is based
 * on air voxel values, position of the segment with respect
 * to the body bounds, size of the segment, position of the
 * segment in the entire volume, relation of the segment with
 * respect to another segments
 *
 * Attributes:
 *      subsample_factor: factor by which volume would be
 *          sub-sampled for certain processing steps in order
 *          to reduce computational time
 *      lung_volume_threshold: lower threshold for both lungs
 *          in mm^3
 *      air_threshold: threshold below which everything is
 *          considered to be air
 *      lung_assumed_center_n: assumed normalized mass center
 *          of the both lungs together
 *      body_bounds_th: side, front and back body bounds thresholds
 *
 * Methods:
 *      LiTS_lung_detector: default and one constructor with all
 *          parameters
 *      lung_segmentation: method for extraction of the binary mask
 *          corresponding to the both lung wings
 */
class LiTS_lung_detector
{

private:

	unsigned int subsample_factor[3] = {8, 8, 1};
	float lung_volume_threshold;
	float air_threshold;
	float lung_assumed_center_n[3] = {0.5, 0.6, 0.9};
	unsigned int body_bounds_th[3] = {20, 20, 50};


public:
	LiTS_lung_detector(float lung_volume_threshold_ = 100.0 * 100.0 * 50.0,
			           float air_threshold_=-0.49);
	LiTS_lung_detector(unsigned int *subsample_factor_,
			           float lung_volume_threshold_,
			           float air_threshold_,
			           float *lung_assumed_center_n_,
			           unsigned int *body_bounds_th_);

	void lung_segmentation(LiTS_scan *scan);

};


#endif /* LUNG_DETECTOR_H_ */
