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

/******************************************************************************
/* LiTS_processor a class for volume and segmentation pre and post processing.
 *
 * It has means of volume intensity normalization, data re-orientation and
 * axis re-ordering; getting default order and orientation of the axes
 *
 * Attributes:
 *
 * 		lower_threshold: lower limit for the voxel intensities
 * 		upper_threshold: upper limit for the voxel intensities
 *
 * 		minimum_value: minimum voxel intensity value in the
 * 			normalized voxel range
 * 		maximum_value: maximum voxel intensity value in the
 * 			normalized voxel range
 *
 * 		approach: selection between "itk" (cpu) and "cuda" (gpu)
 * 			normalization
 *
 * 		orient: orientation of the axes
 * 		order: order of the axes
 *
 * Methods:
 *
 * 		LiTS_preprocessor: constructor
 *
 * 		preprocess_volume: normalize voxel intensities and reorder axes and/or
 * 		    flip volume and segmentation if necessary
 *
 * 		normalize_volume: normalize voxel intensities
 *
 * 		reorient_volume: re-order and/or flip volume axes
 * 		reorient_segmentation: re-order and/or flip segmentation axes
 *
 * 		get_axes_orientation: get the orientation of the axes
 * 		get_axes_order: get the order of the axes
 *
 ******************************************************************************/

class LiTS_processor
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

    LiTS_processor(float lt = -300, float ut = 700.0, float min = -0.5,
                   float max = 0.5, std::string approach = "cuda");

    void preprocess_volume(LiTS_scan *scan);

    void normalize_volume(LiTS_scan *scan);

    void reorient_volume(LiTS_scan *scan,
                         unsigned *cord, short *corient,
                         unsigned *dord, short *dorient);
    void reorient_volume(float *volume,
                         unsigned w, unsigned h, unsigned d,
                         unsigned *cord, short *corient,
                         unsigned *dord, short *dorient);

    void reorient_segmentation(LiTS_scan *scan,
                               unsigned *cord, short *corient,
                               unsigned *dord, short *dorient);
    void reorient_segmentation(unsigned char *segmentation,
                               unsigned w, unsigned h, unsigned d,
                               unsigned *cord, short *corient,
                               unsigned *dord, short *dorient);

    short* get_axes_orientation();
    unsigned* get_axes_order();

};

#endif /* LITS_PRE_AND_POST_PROCESSOR_H_ */
