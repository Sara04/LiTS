
#include "lits_scan.h"


/*
 * LiTS_scan constructor: assigning volume and segmentation
 * 		file paths
 *
 * Arguments:
 * 		volume_path_: path to the volume file
 *		segmentation_path: path to the segmentation (ground truth )file
 */
LiTS_scan::LiTS_scan(std::string volume_path_, std::string segmentation_path_)
{
	volume_path = volume_path_;
	segmentation_path = segmentation_path_;
}

/*
 * load_volume: loading volume from the volume_path
 * 		using volume_reader
 */
void LiTS_scan::load_volume()
{
	volume_reader->SetFileName(volume_path);
	volume_reader->Update();
	volume = volume_reader->GetOutput();
}

/*
 * load_segmentation: loading segmentation from the segmentation_path
 * 		using segmentation_reader
 */
void LiTS_scan::load_segmentation()
{
	segmentation_reader->SetFileName(segmentation_path);
	segmentation_reader->Update();
	segmentation = segmentation_reader->GetOutput();
}

/*
 * load_info: loading volume/segmentation size info and
 * 		verifying their consistency
 * 		(could be extended with another info)
 */
void LiTS_scan::load_info()
{
	VolumeType::RegionType volume_region = volume->GetLargestPossibleRegion();
	SegmentationType::RegionType segment_region = segmentation->GetLargestPossibleRegion();

	VolumeType::SpacingType spacing = volume->GetSpacing();

	VolumeType::SizeType size_v = volume_region.GetSize();
	SegmentationType::SizeType size_s = segment_region.GetSize();

	h = size_v[0];
	w = size_v[1];
	d = size_v[2];

	voxel_h = spacing[0];
	voxel_w = spacing[1];
	voxel_d = spacing[2];

	if(size_v[0] != size_s[0] or size_v[1]!=size_s[1] or size_v[2]!=size_s[2])
	{
		std::cout<<"Volume path:"<<volume_path<<std::endl;
		std::cout<<"Segmentation path:"<<segmentation_path<<std::endl;
		std::cout<<"Volume and segmentation data are not compatible"<<"\n";
	}
}

/*
 * get_volume: returns volume member
 */
VolumeType::Pointer LiTS_scan::get_volume()
{
	return volume;
}

/*
 * set_volume: sets volume member
 * Arguments:
 * 		volume_: pointer to the volume data
 */
void LiTS_scan::set_volume(VolumeType::Pointer volume_)
{
	volume = volume_;
}

/*
 * get_segmentation: returns segmentation member
 */
SegmentationType::Pointer LiTS_scan::get_segmentation()
{
	return segmentation;
}

/*
 * set_segmentation: sets segmentation member
 *
 * Arguments:
 * 		segment_: pointer to the segmentation data
 */
void LiTS_scan::set_segmentation(SegmentationType::Pointer segment_)
{
	segmentation = segment_;
}

/*
 * get_height: returns volume/segmentation height
 */
int LiTS_scan::get_height(){return h;}

/*
 * get_width: returns volume/segmentation width
 */
int LiTS_scan::get_width(){return w;}

/*
 * get_depth: returns volume/segmentation depth
 */
int LiTS_scan::get_depth(){return d;}

/*
 * get_height: returns volume/segmentation height
 */
float LiTS_scan::get_voxel_height(){return voxel_h;}

/*
 * get_width: returns volume/segmentation width
 */
float LiTS_scan::get_voxel_width(){return voxel_w;}

/*
 * get_depth: returns volume/segmentation depth
 */
float LiTS_scan::get_voxel_depth(){return voxel_d;}


/*
 * set_height: sets volume height
 *
 * Arguments:
 * 		h_: height to be set
 */
void LiTS_scan::set_height(int h_){h = h_;}

/*
 * set_width: sets volume width
 *
 * Arguments:
 * 		w_: width to be set
 */
void LiTS_scan::set_width(int w_){w = w_;}

/*
 * set_depth: sets volume depth
 *
 * Arguments:
 * 		d_: depth to be set
 */
void LiTS_scan::set_depth(int d_){d = d_;}
