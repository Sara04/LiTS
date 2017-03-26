#include "lung_segmentator.h"

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
LiTS_lung_segmentator::LiTS_lung_segmentator(float lung_volume_threshold_,
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
LiTS_lung_segmentator::LiTS_lung_segmentator(unsigned int *subsample_factor_,
                                       float lung_volume_threshold_,
                                       float air_threshold_,
                                       float *lung_assumed_center_n_,
                                       unsigned int *body_bounds_th_)
{
    lung_volume_threshold = lung_volume_threshold_;
    air_threshold = air_threshold_;
    for (unsigned int i = 0; i < 3; i++)
    {
        subsample_factor[i] = subsample_factor_[i];
        lung_assumed_center_n[i] = lung_assumed_center_n_[i];
        body_bounds_th[i] = body_bounds_th_[i];
    }
}

/*
 * lung_segmentation: method for extraction of the binary mask
 *          corresponding to the both lung wings
 *
 * Arguments:
 *      scan: pointer to the LiTS_scan object that contains
 *          pointer to the pre-processed volume array and
 *          extracted info about voxel and volume size
 *
 */
void LiTS_lung_segmentator::lung_segmentation(LiTS_scan *scan)
{

    unsigned int assumed_hw = 512;

    subsample_factor[0] = 4;
    subsample_factor[1] = 4;
    subsample_factor[2] = 1;

    float lung_volume_th_vox = lung_volume_threshold /
                               (scan->get_voxel_height() *
                                scan->get_voxel_width() *
                                scan->get_voxel_depth());

    unsigned int size[3] = {scan->get_width(),
                            scan->get_height(),
                            scan->get_depth()};

    if(scan->get_width() > scan->get_height())
        subsample_factor[1] = int(float(subsample_factor[1] *
                                        scan->get_height()) /
                                  scan->get_width());
    if(!subsample_factor[1])
        subsample_factor[1] = 1;

    if(scan->get_width() < scan->get_height())
        subsample_factor[0] = int(float(subsample_factor[1] *
                                        scan->get_width()) /
                                  scan->get_height());
    if(!subsample_factor[0])
        subsample_factor[0] = 1;

    if(size[0] != assumed_hw)
    {
        body_bounds_th[1] = int(float(body_bounds_th[1] * size[0]) /
                                assumed_hw);
        body_bounds_th[2] = int(float(body_bounds_th[2] * size[0]) /
                                assumed_hw);
    }

    if(size[1] != assumed_hw)
        body_bounds_th[0] = int(float(body_bounds_th[0] * size[1]) /
                                assumed_hw);

    bool *lungs_mask = new bool[size[0] * size[1] * size[2]];
    unsigned int *bounds = new unsigned int[4 * size[2]];

    segment_lungs((scan->get_volume())->GetBufferPointer(), size,
                  lungs_mask, bounds,
                  subsample_factor, lung_assumed_center_n, body_bounds_th,
                  lung_volume_th_vox, air_threshold);

    unsigned char *lungs_mask_3 = (unsigned char *)(lungs_mask);
    for(unsigned int i = 0; i < (size[0] * size[1] * size[2]); i++)
        if(!lungs_mask[i])
            lungs_mask_3[i] = 0;
        else
            lungs_mask_3[i] = 3;
    scan->set_lungs_segmentation(lungs_mask_3);
}
