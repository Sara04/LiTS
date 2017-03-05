/*
 * lits_preprocessor.h
 *
 *  Created on: Feb 19, 2017
 *      Author: sara
 */

#ifndef LITS_PREPROCESSOR_H_
#define LITS_PREPROCESSOR_H_

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
class LiTS_preprocessor
{

private:
    float lower_threshold;
    float upper_threshold;
    float minimum_value;
    float maximum_value;
    std::string approach;

public:
    LiTS_preprocessor(float lt = -300, float ut = 700.0, float min = -0.5,
                      float max = 0.5, std::string approach = "cuda");
    void preprocess(LiTS_scan *scan);
};

#endif /* LITS_PREPROCESSOR_H_ */
