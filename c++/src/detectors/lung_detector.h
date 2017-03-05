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
