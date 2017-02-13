/*
 * lits_scan.h
 *
 *  Created on: Feb 11, 2017
 *      Author: sara
 */

#ifndef LITS_SCAN_H_
#define LITS_SCAN_H_
#include <string>
#include "itkImage.h"
#include "itkImageFileReader.h"

typedef itk::Image<double, 3> VolumeType;
typedef itk::ImageFileReader<VolumeType> VolumeReaderType;
typedef itk::Image<unsigned char, 3> SegmentationType;
typedef itk::ImageFileReader<SegmentationType> SegmentationReaderType;

class LiTS_scan
{

private:
	std::string volume_path;
	std::string segmentation_path;

	VolumeType::Pointer volume = VolumeType::New();
	SegmentationType::Pointer segmentation = SegmentationType::New();
	SegmentationType::Pointer lungs = SegmentationType::New();

	VolumeReaderType::Pointer volume_reader = VolumeReaderType::New();
	SegmentationReaderType::Pointer segmentation_reader = SegmentationReaderType::New();

	int h;
	int w;
	int d;

public:

	LiTS_scan(std::string volume_path_, std::string segmentation_path_);
	~LiTS_scan();

	void load_volume();
	void load_segmentation();
	void load_info();

};


#endif /* LITS_SCAN_H_ */
