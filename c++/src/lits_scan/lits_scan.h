/*
 * lits_scan.h
 *
 *  Created on: Feb 11, 2017
 *      Author: sara
 */

#ifndef LITS_SCAN_H_
#define LITS_SCAN_H_

#include <iostream>
#include <string>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

typedef itk::Image<float, 3> VolumeType;
typedef itk::ImageFileReader<VolumeType> VolumeReaderType;
typedef itk::Image<unsigned char, 3> SegmentationType;
typedef itk::ImageFileReader<SegmentationType> SegmentationReaderType;
typedef itk::ImageFileWriter<SegmentationType> SegmentationWriterType;

/* LiTS_scan class for volume and segmentation file management.
 * It has means of volume and segmentation nii file reading,
 * information extraction (height, width, depth);
 * getting data pointer and setting pointers on new data;
 * getting and setting size of the data.
 *
 * Attributes:
 * 		volume_path: path to the file containing volume
 * 		segmentation_path: path to the file containing segmentation
 * 			(ground truth)
 * 		volume: pointer to VolumeType that contains volume data and
 * 			all the information normally provided in nifti format
 * 			(!!! some information fields are not reliable !!!)
 * 		segmentation: pointer to SegmentationType that contains volume
 * 			segmentation (ground truth for liver and tumor) and
 * 			all the information normally provided in nifti format
 * 			(!!! some information fields are not reliable !!!)
 * 		lungs: pointer to SegmentationType that contains detected lungs
 * 		volume_reader: pointer to VolumeReaderType, used for volume
 * 			file reading
 * 		segmentation_reader: pointer to SegmentationReaderType, used for
 * 			segmentation file reading
 * 		h: volume height (front-back body direction)
 * 		w: volume width (left-right body direction)
 * 		d: volume depth (bottom-top body direction)
 * 		h_voxel: voxel height (front-back body direction)
 * 		w_voxel: voxel width (left-right body direction)
 * 		d_voxel: voxel depth (bottom-top body direction)
 * 		axes_order: order of volume's axes
 * 		axes_orientation: orientation of volume's axes
 *
 * Methods:
 * 		LiTS_scan: constructor
 * 		load_volume: loading volume file using volume_reader
 * 		load_segmentation: loading segmentation file using segmentation
 * 			reader
 * 		load_info: loading volume/segmentation size info (could be extended)
 *
 * 		//Getters
 * 		get_volume: get volume member (pointer to the volume data)
 * 		get_segmentation: get segmentation member
 * 			(pointer to the segmentation data)
 * 		get_height: get height of the volume/segmentation
 * 			(front-back body direction)
 * 		get_width: get width of the volume/segmentation
 * 			(left-right body direction)
 * 		get_depth: get depth of the volume/segmentation
 * 			(bottom-top body direction)
 * 		get_voxel_height: get height of the voxels
 * 		get_voxel_width: get width of the voxels
 * 		get_voxel_depth: get depth of the voxels (slice distance)
 * 		get_axes_order: get the order of axes
 * 		get_axes_orientation: get the orientations of axes
 *
 * 		//Setters
 * 		set_volume: set volume member (set pointer to the volume data)
 * 		set_segmentation: set segmentation member
 * 			(set pointer to the segmentation data)
 * 		set_height: set height of the volume/segmentation
 * 			(front-back body direction)
 * 		set_width: set width of the volume/segmentation
 * 			(left-right body direction)
 * 		set_depth: set depth of the volume/segmentation
 * 			(bottom-top body direction)
 * 		set_axes_order: set the order of axes
 * 		set_axes_orientation: set the orientation of axes
 *
 *		//Writers/savers to be done...
 */
class LiTS_scan
{

private:

    std::string volume_path;
    std::string segmentation_path;

    VolumeType::Pointer volume;
    SegmentationType::Pointer segmentation;
    SegmentationType::Pointer lungs_segmentation;
    SegmentationType::Pointer liver_segmentation;
    SegmentationType::Pointer liver_tumor_segmentation;

    VolumeReaderType::Pointer volume_reader;
    SegmentationReaderType::Pointer segmentation_reader;

    int h;
    int w;
    int d;

    float voxel_h;
    float voxel_w;
    float voxel_d;

    unsigned int axes_order[3];
    short axes_orientation[3];

public:

    LiTS_scan(std::string volume_path_, std::string segmentation_path_);
    LiTS_scan(std::string volume_path_);

    void load_volume();
    void load_segmentation();
    void load_info();
    void load_lungs_segmentation(std::string lungs_segmentation_path);
    void load_liver_segmentation(std::string liver_segmentation_path);
    void load_liver_tumor_segmentation(std::string
                                       liver_tumor_segmentation_path);

    void set_volume(VolumeType::Pointer volume_);
    void set_segmentation(SegmentationType::Pointer segment_);
    void set_lungs_segmentation(SegmentationType::Pointer lungs_segment_);
    void set_lungs_segmentation(unsigned char *lungs_segment_);
    void set_liver_segmentation(SegmentationType::Pointer liver_segment_);
    void set_liver_segmentation(unsigned char *liver_segment_);
    void set_liver_tumor_segmentation(SegmentationType::Pointer
                                      liver_tumor_segment_);
    void set_liver_tumor_segmentation(unsigned char *liver_tumor_segment_);

    VolumeType::Pointer get_volume();
    SegmentationType::Pointer get_segmentation();
    SegmentationType::Pointer get_lungs_segmentation();
    SegmentationType::Pointer get_liver_segmentation();
    SegmentationType::Pointer get_liver_tumor_segmentation();

    int get_height();
    int get_width();
    int get_depth();

    float get_voxel_height();
    float get_voxel_width();
    float get_voxel_depth();

    unsigned int * get_axes_order();
    short int * get_axes_orientation();

    void save_lungs_segmentation(std::string lungs_segmentation_path);
    void save_liver_segmentation(std::string liver_segmentation_path);
    void save_liver_tumor_segmentation(std::string
                                       liver_tumor_segmentation_path);

};

#endif /* LITS_SCAN_H_ */
