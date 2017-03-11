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

typedef itk::Image<float, 3> VolumeType;
typedef itk::ImageFileReader<VolumeType> VolumeReaderType;
typedef itk::Image<unsigned char, 3> SegmentationType;
typedef itk::ImageFileReader<SegmentationType> SegmentationReaderType;

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
    bool *lungs_mask;

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
    ~LiTS_scan();

    void load_volume();
    void load_segmentation();
    void load_info();

    VolumeType::Pointer get_volume();
    void set_volume(VolumeType::Pointer volume_);
    SegmentationType::Pointer get_segmentation();
    void set_segmentation(SegmentationType::Pointer segment_);
    bool * get_lungs_mask();
    void set_lungs_mask(bool *lungs_mask_);

    int get_height();
    int get_width();
    int get_depth();

    float get_voxel_height();
    float get_voxel_width();
    float get_voxel_depth();

    unsigned int * get_axes_order();
    short int * get_axes_orientation();

    void set_height(int h_);
    void set_width(int w_);
    void set_depth(int d_);

    void set_axes_order(unsigned int *order);
    void set_axes_orientation(short *orientation);

};

#endif /* LITS_SCAN_H_ */
