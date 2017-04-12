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

/******************************************************************************
/* LiTS_scan a  class for volume and segmentation file management.
 *
 * It has means of volume and segmentation nii file reading,
 * information extraction (height, width, depth);
 * voxel size extraction;
 * getting data pointer and setting pointers on new data;
 * getting and setting size of the data.
 *
 * Attributes:
 *
 * 		volume_path: path to the file that contains volume
 * 		segmentation_path: path to the file that contains segmentation
 * 			(ground truth)
 * 		meta_segmentation_path: path where is/will be meta segmentation
 * 		    stored
 *
 * 		volume: pointer to VolumeType that contains volume data and
 * 			all the information normally provided in nifti format
 * 			(!!! some information fields are not reliable !!!)
 * 		segmentation: pointer to SegmentationType that contains volume
 * 			segmentation (ground truth for liver and tumor) and
 * 			all the information normally provided in nifti format
 * 			(!!! some information fields are not reliable !!!)
 * 		meta_segmentation: pointer to SegmentationType that contains
 * 		    meta segmentation of lungs, liver and tumor necessary for
 * 		    the development/training
 *
 *      volume_reader: pointer to the reader of the VolumeType data
 *      segmentation_reader: pointer to the reader of the SegmentationType data
 *
 * 		h: volume height (front-back body direction)
 * 		w: volume width (left-right body direction)
 * 		d: volume depth (bottom-top body direction)
 *
 * 		h_voxel: voxel height (front-back body direction)
 * 		w_voxel: voxel width (left-right body direction)
 * 		d_voxel: voxel depth (bottom-top body direction)
 *
 * 		axes_order: order of the volume's axes
 * 		axes_orientation: orientation of the volume's axes
 *
 * Methods:
 *
 * 		LiTS_scan: constructor
 *
 *      //Loaders
 * 		load_volume: loading volume file using volume_reader
 * 		load_segmentation: loading segmentation file using segmentation
 * 			reader
 * 		load_info: loading volume/segmentation info
 * 		load_meta_segmentation: loading meta segmentation file using
 * 		    segmentation reader
 *
 * 		//Getters
 * 		get_volume: get pointer to the volume data
 * 		get_segmentation: get pointer to the segmentation data
 * 	    get_meta_segmentaion: get pointer to the meta segmentation data
 *
 * 		get_height: get height of the volume/segmentation
 * 			(front-back body direction)
 * 		get_width: get width of the volume/segmentation
 * 			(left-right body direction)
 * 		get_depth: get depth of the volume/segmentation
 * 			(bottom-top body direction)
 *
 * 		get_voxel_height: get height of the voxels
 * 		get_voxel_width: get width of the voxels
 * 		get_voxel_depth: get depth of the voxels (slice distance)
 *
 * 		get_axes_order: get the order of the axes
 * 		get_axes_orientation: get the orientations of the axes
 *
 * 		//Setters
 * 		set_volume: set pointer to the volume data
 * 		set_segmentation: set pointer to the segmentation data
 * 		set_meta_segmentation: set pointer to the meta segmentation data
 *
 * 		//Savers
 * 		save_meta_segmentation: save meta segmentation in nii format
 * 		    in given file
 * 		save_tumor_segmentation: save tumor segmentation in nii format
 * 		    in given file
 *
 *****************************************************************************/
class LiTS_scan
{

private:

    std::string volume_path;
    std::string segmentation_path;
    std::string meta_segmentation_path;

    VolumeType::Pointer volume;
    SegmentationType::Pointer segmentation;
    SegmentationType::Pointer meta_segmentation;

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
    void load_meta_segmentation();

    void set_volume(VolumeType::Pointer volume_);
    void set_segmentation(SegmentationType::Pointer segment_);
    void set_meta_segmentation(SegmentationType::Pointer lungs_segment_);
    void set_meta_segmentation(unsigned char *lungs_segment_);
    void set_meta_segmentation(bool *segment, unsigned len, unsigned char v);

    VolumeType::Pointer get_volume();
    SegmentationType::Pointer get_segmentation();
    SegmentationType::Pointer get_meta_segmentation();

    int get_height();
    int get_width();
    int get_depth();

    float get_voxel_height();
    float get_voxel_width();
    float get_voxel_depth();

    unsigned int * get_axes_order();
    short int * get_axes_orientation();

    void save_meta_segmentation(std::string meta_segmentation_path_);

};

#endif /* LITS_SCAN_H_ */
