/*
 * lits_preprocessor.h
 *
 *  Created on: Feb 19, 2017
 *      Author: sara
 */

#ifndef LITS_PRE_AND_POST_PROCESSOR_H_
#define LITS_PRE_AND_POST_PROCESSOR_H_

#include "../lits_scan/lits_scan.h"
#include <string>
#include <itkImage.h>
#include <itkIntensityWindowingImageFilter.h>
#include <itkOrientImageFilter.h>

typedef itk::IntensityWindowingImageFilter<VolumeType, VolumeType>
        RescalerType;
typedef itk::OrientImageFilter<VolumeType, VolumeType> OrientVolumeType;
typedef itk::OrientImageFilter<SegmentationType, SegmentationType>
        OrientSegmentationType;

/* LiTS_preprocessor class for volume voxel intensity normalization
 * and volume and segmentation re-orienting to RAS coordinate
 * system
 *
 * Attributes:
 * 		lower_threshold: lower limit for voxel intensity
 * 		upper_threshold: upper limit for voxel intensity
 * 		minimum_value: minimum voxel intensity value in the
 * 			normalized voxel range
 * 		maximum_value: maximum voxel intensity value in the
 * 			normalized voxel range
 * 		approach: selection between "itk" (cpu) and "cuda" (gpu)
 * 			normalization
 *
 * Methods:
 * 		LiTS_preprocessor: constructor
 * 		preprocess: normalize voxel intensities and flip volume and
 * 			segmentation if necessary
 */
class LiTS_pre_and_post_processor
{

private:
    float lower_threshold;
    float upper_threshold;
    float minimum_value;
    float maximum_value;
    std::string approach;
    short orient[3];
    unsigned ord[3];

public:

    LiTS_pre_and_post_processor(float lt = -300, float ut = 700.0, float min = -0.5,
                                float max = 0.5, std::string approach = "cuda");

    void preprocess_volume(LiTS_scan *scan);

    void normalize_volume(LiTS_scan *scan);

    void reorient_volume(LiTS_scan *scan,
                         unsigned *cord, short *corient,
                         unsigned *dord, short *dorient);

    void reorient_segmentation(LiTS_scan *scan,
                               unsigned *cord, short *corient,
                               unsigned *dord, short *dorient);

    void reorient_segmentation(unsigned char *segmentation,
                               unsigned w, unsigned h, unsigned d,
                               unsigned *cord, short *corient,
                               unsigned *dord, short *dorient);

    short * get_axes_orientation();
    unsigned *get_axes_order();

};

#endif /* LITS_PRE_AND_POST_PROCESSOR_H_ */
