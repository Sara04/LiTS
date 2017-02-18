/*
 * preprocessor_itk.h
 *
 *  Created on: Feb 14, 2017
 *      Author: sara
 */

#ifndef PREPROCESSOR_ITK_H_
#define PREPROCESSOR_ITK_H_

#include "itkImage.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkOrientImageFilter.h"
#include "../lits_scan/lits_scan.h"

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

	RescalerType::Pointer rescaler;

public:
	LiTS_preprocessor(float lt=-300, float ut=700.0, float min=-0.5, float max=0.5);
	~LiTS_preprocessor();
	void preprocess(LiTS_scan *scan);
};




#endif /* PREPROCESSOR_ITK_H_ */
