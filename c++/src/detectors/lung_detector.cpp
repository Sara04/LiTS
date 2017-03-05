
#include "lung_detector.h"

/*
 * LiTS_lung_detector constructor: initialization of
 * 		attributes
 *
 * Arguments:
 * 		lung_volume_threshold_: assumed lung volume in mm^3
 * 		air_threshold: threshold below which voxel intensities
 * 			are considered to belong to the air
 *
 */
LiTS_lung_detector::LiTS_lung_detector(float lung_volume_threshold_,
                                       float air_threshold_)
{
	lung_volume_threshold = lung_volume_threshold_;
	air_threshold = air_threshold_;
}

/*
 * LiTS_lung_detector constructor: initialization of
 * 		attributes
 *
 * Arguments:
 * 		subsample_factor_: sub-sampling factor for each axis
 * 		    by which volume would be down-sampled in order
 * 		    to reduce computation time
 * 		lung_volume_threshold_: assumed lung volume in mm^3
 * 		air_threshold: threshold below which voxel intensities
 * 			are considered to belong to the air
 * 		lung_assumed_center_n_: assumed center of the lungs mask
 * 			normalized with respect to the volume size along
 * 			each axis
 * 		body_bounds_th_: thresholds used for side, front and back
 * 			body bounds detection
 *
 */
LiTS_lung_detector::LiTS_lung_detector(unsigned int *subsample_factor_,
                                       float lung_volume_threshold_,
                                       float air_threshold_,
                                       float *lung_assumed_center_n_,
                                       unsigned int *body_bounds_th_)
{
	lung_volume_threshold = lung_volume_threshold_;
	air_threshold = air_threshold_;
	for(unsigned int i = 0; i < 3; i++)
	{
		subsample_factor[i] = subsample_factor_[i];
		lung_assumed_center_n[i] = lung_assumed_center_n_[i];
		body_bounds_th[i] = body_bounds_th_[i];
	}
}

void LiTS_lung_detector::lung_segmentation(LiTS_scan *scan)
{

	float lung_volume_th_vox = lung_volume_threshold /
			                   (scan->get_voxel_height() *
			                    scan->get_voxel_width() *
			                    scan->get_voxel_depth());
	unsigned int size[] = {scan->get_width(),
                           scan->get_height(),
                           scan->get_depth()};
	segment_lungs((scan->get_volume())->GetBufferPointer(),
				  size,
			      subsample_factor,
			      lung_assumed_center_n,
			      body_bounds_th,
			      lung_volume_th_vox,
			      air_threshold);
}
