
#include "lits_scan.h"

LiTS_scan::LiTS_scan(std::string volume_path_, std::string segmentation_path_)
{
	volume_path = volume_path_;
	segmentation_path = segmentation_path_;
}

LiTS_scan::~LiTS_scan()
{
	volume_path = "";
	segmentation_path = "";
	h = 0;
	w = 0;
	d = 0;
}

void LiTS_scan::load_volume()
{
	volume_reader->SetFileName(volume_path);
	volume_reader->Update();
	volume = volume_reader->GetOutput();
}

void LiTS_scan::load_segmentation()
{
	segmentation_reader->SetFileName(segmentation_path);
	segmentation_reader->Update();
	segmentation = segmentation_reader->GetOutput();
}

void LiTS_scan::load_info()
{
	VolumeType::RegionType volume_region = volume->GetLargestPossibleRegion();
	SegmentationType::RegionType segment_region = segmentation->GetLargestPossibleRegion();

	VolumeType::SizeType size_v = volume_region.GetSize();
	SegmentationType::SizeType size_s = segment_region.GetSize();

	h = size_v[0];
	w = size_v[1];
	d = size_v[2];

	if(size_v[0] != size_s[0] or size_v[1]!=size_s[1] or size_v[2]!=size_s[2])
	{
		std::cout<<"Volume path:"<<volume_path<<std::endl;
		std::cout<<"Segmentation path:"<<segmentation_path<<std::endl;
		std::cout<<"Volume and segmentation data are not compatible"<<"\n";
	}
}

VolumeType::Pointer LiTS_scan::get_volume()
{
	return volume;
}

void LiTS_scan::set_volume(VolumeType::Pointer volume_)
{
	volume = volume_;
}

