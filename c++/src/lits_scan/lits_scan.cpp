#include "lits_scan.h"

/******************************************************************************
 * LiTS_scan constructor: assigning volume and segmentation file paths
 *
 * Arguments:
 * 		volume_path_: path to the volume file
 *		segmentation_path: path to the segmentation (ground truth) file
 *****************************************************************************/
LiTS_scan::LiTS_scan(std::string volume_path_, std::string segmentation_path_)
{
    volume = VolumeType::New();
    segmentation = SegmentationType::New();

    volume_path = volume_path_;
    segmentation_path = segmentation_path_;

    for(unsigned int i = 0; i < 3; i++)
    {
        axes_order[i] = i;
        axes_orientation[i] = 1;
    }
}

/******************************************************************************
 * LiTS_scan constructor: assigning volume file path
 *
 * Arguments:
 * 		volume_path_: path to the volume file
 *****************************************************************************/
LiTS_scan::LiTS_scan(std::string volume_path_)
{
    volume = VolumeType::New();

    volume_path = volume_path_;

    for(unsigned int i = 0; i < 3; i++)
    {
        axes_order[i] = i;
        axes_orientation[i] = 1;
    }
}

/******************************************************************************
 * load_volume: loading volume from the volume_path using volume_reader
 *****************************************************************************/
void LiTS_scan::load_volume()
{
    VolumeReaderType::Pointer volume_reader = VolumeReaderType::New();
    volume_reader->SetFileName(volume_path);
    volume_reader->Update();
    volume = volume_reader->GetOutput();
}

/******************************************************************************
 * load_segmentation: loading segmentation from the segmentation_path using
 * segmentation_reader
 *****************************************************************************/
void LiTS_scan::load_segmentation()
{
    if (segmentation.IsNotNull())
    {
        SegmentationReaderType::Pointer segmentation_reader =
                SegmentationReaderType::New();
        segmentation_reader->SetFileName(segmentation_path);
        segmentation_reader->Update();
        segmentation = segmentation_reader->GetOutput();
    }
    else
    {
        std::cout<<"Segmentation path does not exist!"<<std::endl;
        exit (EXIT_FAILURE);
    }
}

/******************************************************************************
 * load_info: loading volume/segmentation info - size, spacing, orientation,
 * order of axes
 *****************************************************************************/
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
                axes_order[i] = j;
        }
        axes_orientation[axes_order[i]] = direction_v[i][axes_order[i]];
    }

    w = size_v[axes_order[0]];
    h = size_v[axes_order[1]];
    d = size_v[axes_order[2]];

    voxel_w = spacing[axes_order[0]];
    voxel_h = spacing[axes_order[1]];
    voxel_d = spacing[axes_order[2]];

    if (segmentation.IsNotNull())
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
            std::cout<<"Segmentation is not loaded"<<std::endl;
            exit (EXIT_FAILURE);
        }
    }
}

/******************************************************************************
 * load_lungs_segmentation: loading lungs segmentation from the
 * lungs_segmentation_path using segmentation_reader
 * Arguments:
 *      lungs_segmentation_path: path to the lungs segmentation file
 *****************************************************************************/
void LiTS_scan::load_lungs_segmentation(std::string lungs_segmentation_path)
{


    if (fs::exists(lungs_segmentation_path))
    {
        SegmentationReaderType::Pointer segmentation_reader =
                SegmentationReaderType::New();
        lungs_segmentation = SegmentationType::New();
        segmentation_reader->SetFileName(lungs_segmentation_path);
        segmentation_reader->Update();
        lungs_segmentation = segmentation_reader->GetOutput();
    }
    else
    {
        std::cout<<"Lungs segmentation path does not exist!"<<std::endl;
        exit (EXIT_FAILURE);
    }
}

/******************************************************************************
 * load_liver_segmentation: loading liver segmentation from the
 * liver_segmentation_path using segmentation_reader
 * Arguments:
 *      liver_segmentation_path: path to the liver segmentation file
 *****************************************************************************/
void LiTS_scan::load_liver_segmentation(std::string liver_segmentation_path)
{

    if (fs::exists(liver_segmentation_path))
    {
        SegmentationReaderType::Pointer segmentation_reader =
                SegmentationReaderType::New();
        liver_segmentation = SegmentationType::New();
        segmentation_reader->SetFileName(liver_segmentation_path);
        segmentation_reader->Update();
        liver_segmentation = segmentation_reader->GetOutput();
    }
    else
    {
        std::cout<<"Liver segmentation path does not exist!"<<std::endl;
        exit (EXIT_FAILURE);
    }
}

/******************************************************************************
 * load_liver_tumor_segmentation: loading liver tumor segmentation from the
 * liver_tumor_segmentation_path using segmentation_reader
 * Arguments:
 *      liver_tumor_segmentation_path: path to the liver tumor segmentation
 *      file
 *****************************************************************************/
void LiTS_scan::load_liver_tumor_segmentation(std::string
                                              liver_tumor_segmentation_path)
{

    if (fs::exists(liver_tumor_segmentation_path))
    {
        SegmentationReaderType::Pointer segmentation_reader =
                SegmentationReaderType::New();
        liver_tumor_segmentation = SegmentationType::New();
        segmentation_reader->SetFileName(liver_tumor_segmentation_path);
        segmentation_reader->Update();
        liver_tumor_segmentation = segmentation_reader->GetOutput();
    }
    else
    {
        std::cout<<"Liver tumor segmentation path does not exist!"<<std::endl;
        exit (EXIT_FAILURE);
    }
}

/******************************************************************************
 * set_volume: sets volume member
 * Arguments:
 *      volume_: pointer to the volume data
 *****************************************************************************/
void LiTS_scan::set_volume(VolumeType::Pointer volume_)
{
    volume = volume_;
}

/******************************************************************************
 * set_segmentation: sets segmentation member
 * Arguments:
 *      segment_: pointer to the segmentation data
 *****************************************************************************/
void LiTS_scan::set_segmentation(SegmentationType::Pointer segment_)
{
    segmentation = segment_;
}

/******************************************************************************
 * set_lungs_segmentation: sets lungs_segmentation member
 * Arguments:
 *      lungs_segment_: pointer to the lungs segmentation data
 *****************************************************************************/
void LiTS_scan::set_lungs_segmentation(SegmentationType::Pointer
                                       lungs_segment_)
{
    lungs_segmentation = lungs_segment_;
}

/******************************************************************************
 * set_lungs_segmentation: creates and sets lungs_segmentation member
 * Arguments:
 *      lungs_segment_: buffer containing lungs segmentation
 *****************************************************************************/
void LiTS_scan::set_lungs_segmentation(unsigned char *lungs_segment)
{
    lungs_segmentation = SegmentationType::New();

    SegmentationType::IndexType desired_start;
    SegmentationType::SizeType desired_size;
    SegmentationType::SpacingType spacing;

    desired_start[0] = 0;
    desired_start[0] = 0;
    desired_start[0] = 0;

    desired_size[axes_order[0]] = w;
    desired_size[axes_order[1]] = h;
    desired_size[axes_order[2]] = d;

    SegmentationType::RegionType desiredRegion(desired_start, desired_size);

    lungs_segmentation->SetRegions(desiredRegion);
    lungs_segmentation->Allocate();

    spacing[axes_order[0]] = voxel_w;
    spacing[axes_order[1]] = voxel_h;
    spacing[axes_order[2]] = voxel_d;
    lungs_segmentation->SetSpacing(spacing);

    unsigned int segment_size = w * h * d;
    memcpy(lungs_segmentation->GetBufferPointer(), lungs_segment, segment_size);
}

/******************************************************************************
 * set_liver_segmentation: sets liver_segmentation member
 * Arguments:
 *      liver_segment_: pointer to the segmentation data
 *****************************************************************************/
void LiTS_scan::set_liver_segmentation(SegmentationType::Pointer
                                       liver_segment_)
{
    liver_segmentation = liver_segment_;
}

/******************************************************************************
 * set_liver_segmentation: creates and sets liver_segmentation member
 * Arguments:
 *      liver_segment_: buffer containing liver segmentation
 *****************************************************************************/
void LiTS_scan::set_liver_segmentation(unsigned char *liver_segment)
{
    liver_segmentation = SegmentationType::New();

    SegmentationType::IndexType desired_start;
    SegmentationType::SizeType desired_size;
    SegmentationType::SpacingType spacing;

    desired_start[0] = 0;
    desired_start[0] = 0;
    desired_start[0] = 0;

    desired_size[axes_order[0]] = w;
    desired_size[axes_order[1]] = h;
    desired_size[axes_order[2]] = d;

    SegmentationType::RegionType desiredRegion(desired_start, desired_size);

    liver_segmentation->SetRegions(desiredRegion);
    liver_segmentation->Allocate();

    spacing[axes_order[0]] = voxel_w;
    spacing[axes_order[1]] = voxel_h;
    spacing[axes_order[2]] = voxel_d;
    liver_segmentation->SetSpacing(spacing);

    unsigned int segment_size = w * h * d;
    memcpy(liver_segmentation->GetBufferPointer(), liver_segment, segment_size);
}

/******************************************************************************
 * set_liver_tumor_segmentation: sets liver_tumor_segmentation member
 * Arguments:
 *      liver_tumor_segment_: pointer to the liver tumor segmentation data
 *****************************************************************************/
void LiTS_scan::set_liver_tumor_segmentation(SegmentationType::Pointer
                                             liver_tumor_segment_)
{
    liver_tumor_segmentation = liver_tumor_segment_;
}

/******************************************************************************
 * set_liver_segmentation: creates and sets liver_tumor_segmentation member
 * Arguments:
 *      liver_tumor_segment_: buffer containing liver tumor segmentation
 *****************************************************************************/
void LiTS_scan::set_liver_tumor_segmentation(unsigned char *
                                             liver_tumor_segment)
{
    liver_tumor_segmentation = SegmentationType::New();

    SegmentationType::IndexType desired_start;
    SegmentationType::SizeType desired_size;
    SegmentationType::SpacingType spacing;

    desired_start[0] = 0;
    desired_start[0] = 0;
    desired_start[0] = 0;

    desired_size[axes_order[0]] = w;
    desired_size[axes_order[1]] = h;
    desired_size[axes_order[2]] = d;

    SegmentationType::RegionType desiredRegion(desired_start, desired_size);

    liver_tumor_segmentation->SetRegions(desiredRegion);
    liver_tumor_segmentation->Allocate();

    spacing[axes_order[0]] = voxel_w;
    spacing[axes_order[1]] = voxel_h;
    spacing[axes_order[2]] = voxel_d;
    liver_tumor_segmentation->SetSpacing(spacing);

    unsigned int segment_size = w * h * d;
    memcpy(liver_tumor_segmentation->GetBufferPointer(),
           liver_tumor_segment, segment_size);
}

/******************************************************************************
 * get_volume: returns volume member
 *****************************************************************************/
VolumeType::Pointer LiTS_scan::get_volume()
{
    return volume;
}

/******************************************************************************
 * get_segmentation: returns segmentation member
 *****************************************************************************/
SegmentationType::Pointer LiTS_scan::get_segmentation()
{
    return segmentation;
}

/******************************************************************************
 * get_lungs_mask: returns pointer to lungs mask
 *****************************************************************************/
SegmentationType::Pointer LiTS_scan::get_lungs_segmentation()
{
    return lungs_segmentation;
}

/******************************************************************************
 * get_liver_mask: returns pointer to liver mask
 *****************************************************************************/
SegmentationType::Pointer LiTS_scan::get_liver_segmentation()
{
    return liver_segmentation;
}

/******************************************************************************
 * get_liver_tumor_mask: returns pointer to liver tumor mask
 *****************************************************************************/
SegmentationType::Pointer LiTS_scan::get_liver_tumor_segmentation()
{
    return liver_tumor_segmentation;
}

/******************************************************************************
 * get_axes_order: returns pointer to axes order
 *****************************************************************************/
short int * LiTS_scan::get_axes_orientation()
{
    return axes_orientation;
}

/******************************************************************************
 * get_axes_order: returns pointer to axes order
 *****************************************************************************/
unsigned int * LiTS_scan::get_axes_order()
{
    return axes_order;
}

/******************************************************************************
 * get_height: returns volume/segmentation height
 *****************************************************************************/
int LiTS_scan::get_height()
{
    return h;
}

/******************************************************************************
 * get_width: returns volume/segmentation width
 *****************************************************************************/
int LiTS_scan::get_width()
{
    return w;
}

/******************************************************************************
 * get_depth: returns volume/segmentation depth
 *****************************************************************************/
int LiTS_scan::get_depth()
{
    return d;
}

/******************************************************************************
 * get_height: returns volume/segmentation height
 *****************************************************************************/
float LiTS_scan::get_voxel_height()
{
    return voxel_h;
}

/******************************************************************************
 * get_width: returns volume/segmentation width
 *****************************************************************************/
float LiTS_scan::get_voxel_width()
{
    return voxel_w;
}

/******************************************************************************
 * get_depth: returns volume/segmentation depth
 *****************************************************************************/
float LiTS_scan::get_voxel_depth()
{
    return voxel_d;
}

/******************************************************************************
 * save_lungs_segmentation: save lungs_segmentation at input path
 * Arguments:
 *      lungs_segmentation_path:
 *****************************************************************************/
void LiTS_scan::save_lungs_segmentation(std::string lungs_segmentation_path)
{
    SegmentationWriterType::Pointer segmentation_writer =
            SegmentationWriterType::New();

    segmentation_writer->SetFileName(lungs_segmentation_path);
    segmentation_writer->SetInput(lungs_segmentation);
    segmentation_writer->Update();

}
void save_liver_segmentation(std::string liver_segmentation_path);
void save_liver_tumor_segmentation(std::string
                                   liver_tumor_segmentation_path);
