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
    lungs_mask = NULL;

    for(unsigned int i = 0; i < 3; i++)
    {
        axes_order[i] = i;
        axes_orientation[i] = 1;
    }
}

/*
 * LiTS_scan constructor: assigning volume
 * 		file paths
 *
 * Arguments:
 * 		volume_path_: path to the volume file
 */
LiTS_scan::LiTS_scan(std::string volume_path_)
{
    volume_path = volume_path_;
    lungs_mask = NULL;

    for(unsigned int i = 0; i < 3; i++)
    {
        axes_order[i] = i;
        axes_orientation[i] = 1;
    }
}

/*
 * ~LiTS_scan destructor: if memory is allocated
 *      for lungs_mask, release it
 */
LiTS_scan::~LiTS_scan()
{
    if(lungs_mask)
        delete [] lungs_mask;
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
    if (segmentation_path.size())
    {
        segmentation_reader->SetFileName(segmentation_path);
        segmentation_reader->Update();
        segmentation = segmentation_reader->GetOutput();
    }
    else
    {
        std::cout << "Segmentation path does not exist!" << std::endl;
        exit (EXIT_FAILURE);
    }
}

/*
 * load_info: loading volume/segmentation size info and
 * 		verifying their consistency
 * 		(could be extended with another info)
 */
void LiTS_scan::load_info()
{
    VolumeType::RegionType volume_region = volume->GetLargestPossibleRegion();
    VolumeType::SpacingType spacing = volume->GetSpacing();
    VolumeType::SizeType size_v = volume_region.GetSize();
    VolumeType::DirectionType direction_v = volume->GetDirection();

    for(unsigned int i = 0; i < 3; i++)
    {
        for(unsigned int j = 0; j < 3; j++)
        {
            if(direction_v[i][j] != 0)
            {
                axes_order[i] = j;
                axes_orientation[i] = short(direction_v[i][j]);
            }
        }
    }

    h = size_v[axes_order[0]];
    w = size_v[axes_order[1]];
    d = size_v[axes_order[2]];

    voxel_h = spacing[axes_order[0]];
    voxel_w = spacing[axes_order[1]];
    voxel_d = spacing[axes_order[2]];

    std::cout<<"height:"<<h<<std::endl;
    std::cout<<"width:"<<w<<std::endl;
    std::cout<<"no slices:"<<d<<std::endl;

    std::cout<<"voxel height:"<<voxel_h<<std::endl;
    std::cout<<"voxel width:"<<voxel_w<<std::endl;
    std::cout<<"voxel no slices:"<<voxel_d<<std::endl;

    if (segmentation_path.size())
    {
        SegmentationType::RegionType segment_region = segmentation
                ->GetLargestPossibleRegion();
        SegmentationType::SizeType size_s = segment_region.GetSize();

        if (size_s[0])
        {
            if (size_v[0] != size_s[0] or size_v[1] != size_s[1]
                or size_v[2] != size_s[2])
            {
                std::cout << "Volume path:" << volume_path << std::endl;
                std::cout << "Segmentation path:" << segmentation_path
                          << std::endl;
                std::cout << "Volume and segmentation data are not compatible"
                          << "\n";
            }
        }
        else
        {
            std::cout << "Segmentation is not loaded" << std::endl;
            exit (EXIT_FAILURE);
        }
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
 * get_lungs_mask: returns pointer to lungs_mask
 */

bool * LiTS_scan::get_lungs_mask()
{
    return lungs_mask;
}

/*
 * set_lungs_mask: returns pointer to lungs_mask
 */
void LiTS_scan::set_lungs_mask(bool *lungs_mask_)
{
       lungs_mask = lungs_mask_;
}

/*
 * get_axes_order: returns pointer to axes order
 */
unsigned int * LiTS_scan::get_axes_order()
{
    return axes_order;
}

/*
 * set_axes_order: sets order of axes
 */
void LiTS_scan::set_axes_order(unsigned int *order)
{
    for(unsigned int i = 0; i < 3; i++)
        axes_order[i] = order[i];
}

/*
 * get_axes_orientation: returns pointer to axes
 * orientation
 */
short int * LiTS_scan::get_axes_orientation()
{
    return axes_orientation;
}

/*
 * set_axes_orientation: sets orientation of the
 * axes
 */
void LiTS_scan::set_axes_orientation(short *orientation)
{
    for(unsigned int i = 0; i < 3; i++)
        axes_orientation[i] = orientation[i];
}

/*
 * get_height: returns volume/segmentation height
 */
int LiTS_scan::get_height()
{
    return h;
}

/*
 * get_width: returns volume/segmentation width
 */
int LiTS_scan::get_width()
{
    return w;
}

/*
 * get_depth: returns volume/segmentation depth
 */
int LiTS_scan::get_depth()
{
    return d;
}

/*
 * get_height: returns volume/segmentation height
 */
float LiTS_scan::get_voxel_height()
{
    return voxel_h;
}

/*
 * get_width: returns volume/segmentation width
 */
float LiTS_scan::get_voxel_width()
{
    return voxel_w;
}

/*
 * get_depth: returns volume/segmentation depth
 */
float LiTS_scan::get_voxel_depth()
{
    return voxel_d;
}

/*
 * set_height: sets volume height
 *
 * Arguments:
 * 		h_: height to be set
 */
void LiTS_scan::set_height(int h_)
{
    h = h_;
}

/*
 * set_width: sets volume width
 *
 * Arguments:
 * 		w_: width to be set
 */
void LiTS_scan::set_width(int w_)
{
    w = w_;
}

/*
 * set_depth: sets volume depth
 *
 * Arguments:
 * 		d_: depth to be set
 */
void LiTS_scan::set_depth(int d_)
{
    d = d_;
}
