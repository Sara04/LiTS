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

typedef itk::Image<float, 3> VolumeType;
typedef itk::IntensityWindowingImageFilter<VolumeType, VolumeType> RescalerType;
typedef itk::OrientImageFilter<VolumeType, VolumeType> OrientVolumeType;

class LiTS_preprocessor
{
private:
	float lower_threshold;
	float upper_threshold;
	float minimum_value;
	float maximum_value;
	std::string approach;

public:
	LiTS_preprocessor(float lt=-300, float ut=700.0,
			          float min=-0.5, float max=0.5,
			          std::string approach="cuda");
	~LiTS_preprocessor();
	void preprocess(LiTS_scan *scan);
};

#endif /* LITS_PREPROCESSOR_H_ */
