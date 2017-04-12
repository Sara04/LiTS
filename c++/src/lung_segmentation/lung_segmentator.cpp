#include "lung_segmentator.h"

/******************************************************************************
 * LiTS_lung_detector constructor: initialization of
 * 		attributes
 *
 * Arguments:
 * 		lung_volume_threshold_: assumed lung volume in mm^3
 * 		air_threshold: threshold below which voxel intensities
 * 			are considered to belong to the air
 *
 *****************************************************************************/
LiTS_lung_segmentator::LiTS_lung_segmentator(float lung_volume_threshold_,
                                             float air_threshold_)
{
    lung_volume_threshold = lung_volume_threshold_;
    air_threshold = air_threshold_;
}

/******************************************************************************
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
 *****************************************************************************/
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
        ds_factor[i] = subsample_factor_[i];
        lung_assumed_center_n[i] = lung_assumed_center_n_[i];
        body_bounds_th[i] = body_bounds_th_[i];
    }
}

/******************************************************************************
 * adjust_segmentation_parameters: method for segmentation parameter adjusting
 *  with respect to the input volume size
 *
 * Arguments:
 *      scan: pointer to the LiTS_scan object that contains
 *          pointer to the pre-processed volume array and
 *          extracted info about voxel and volume size
 *
 *****************************************************************************/
void LiTS_lung_segmentator::
    adjust_segmentation_parameters(LiTS_scan *scan, unsigned int *ds_factor_a,
                                   float &lung_V_th_vox, unsigned int *bbounds)
{

    lung_V_th_vox = lung_volume_threshold /
                    (scan->get_voxel_height() *
                     scan->get_voxel_width() *
                     scan->get_voxel_depth());
    unsigned int hw;
    for(unsigned int i = 0; i < 3; i++)
        ds_factor_a[i] = ds_factor[i];

    if(scan->get_width() > scan->get_height())
    {
        hw = scan->get_width();
        ds_factor_a[1] = int(float(ds_factor[1] * scan->get_height()) / hw);
    }
    else
    {
        hw = scan->get_height();
        ds_factor_a[0] = int(float(ds_factor[0] * scan->get_width()) / hw);
    }

    if(!ds_factor_a[1])
        ds_factor_a[1] = 1;

    if(!ds_factor_a[0])
        ds_factor_a[0] = 1;

    if(scan->get_width() != hw)
    {
        bbounds[1] = int(float(body_bounds_th[1] * scan->get_width()) / hw);
        bbounds[2] = int(float(body_bounds_th[2] * scan->get_width()) / hw);
    }
    else
    {
        bbounds[1] = body_bounds_th[1];
        bbounds[2] = body_bounds_th[2];
    }

    if(scan->get_height() != hw)
        bbounds[0] = int(float(body_bounds_th[0] * scan->get_height()) / hw);
    else
        bbounds[0] = body_bounds_th[0];
}

/******************************************************************************
 * lung_segmentation: method for extraction of the binary mask
 *          corresponding to the both lung wings
 *
 * Arguments:
 *      scan: pointer to the LiTS_scan object that contains
 *          pointer to the pre-processed volume array and
 *          extracted info about voxel and volume size
 *
 *****************************************************************************/
void LiTS_lung_segmentator::lung_segmentation(LiTS_scan *scan)
{

    // 1. Adjust segmentation parameters
    unsigned int ds_factor_a[3];
    float lung_V_th_vox = 0;
    unsigned int bbounds[3];
    adjust_segmentation_parameters(scan, ds_factor_a, lung_V_th_vox, bbounds);

    // 2. Lung segmentation
    unsigned S[3] = {scan->get_width(), scan->get_height(), scan->get_depth()};
    bool *lungs_mask = new bool[S[0] * S[1] * S[2]];
    segment_lungs((scan->get_volume())->GetBufferPointer(), S, lungs_mask,
                  ds_factor, lung_assumed_center_n, body_bounds_th,
                  lung_V_th_vox, air_threshold);

    // 3. Saving lung segmentation to meta segmentation
    scan->set_meta_segmentation(lungs_mask, S[0] * S[1] * S[2], 3);
}

