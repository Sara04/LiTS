#include "lits_scan.h"

/******************************************************************************
 * LiTS_scan constructor: assigning volume and segmentation file paths,
 *  volume and segmentation member dynamic construction,
 *  initializing axes_order and axes_orientation members
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
 * LiTS_scan constructor: assigning volume file path, volume member dynamic
 * construction, initializing axes_order and axes_orientation members
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
 * load_meta_segmentation: loading meta segmentation from the
 * meta_segmentation_path using segmentation_reader
 *****************************************************************************/
void LiTS_scan::load_meta_segmentation()
{

    if (fs::exists(meta_segmentation_path))
    {
        SegmentationReaderType::Pointer segmentation_reader =
                SegmentationReaderType::New();
        meta_segmentation = SegmentationType::New();
        segmentation_reader->SetFileName(meta_segmentation_path);
        segmentation_reader->Update();
        meta_segmentation = segmentation_reader->GetOutput();
    }
    else
    {
        std::cout<<"Meta segmentation path does not exist!"<<std::endl;
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
 * set_meta_segmentation: sets meta_segmentation member
 * Arguments:
 *      meta_segment_: pointer to the meta segmentation data
 *****************************************************************************/
void LiTS_scan::set_meta_segmentation(SegmentationType::Pointer
                                       meta_segment_)
{
    meta_segmentation = meta_segment_;
}


/******************************************************************************
 * set_meta_segmentation: constructs and sets meta_segmentation member
 * Arguments:
 *      meta_segment_: buffer containing lungs segmentation
 *****************************************************************************/
void LiTS_scan::set_meta_segmentation(unsigned char *meta_segment_)
{
    meta_segmentation = SegmentationType::New();

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

    meta_segmentation->SetRegions(desiredRegion);
    meta_segmentation->Allocate();

    spacing[axes_order[0]] = voxel_w;
    spacing[axes_order[1]] = voxel_h;
    spacing[axes_order[2]] = voxel_d;
    meta_segmentation->SetSpacing(spacing);

    unsigned int segment_size = w * h * d;
    memcpy(meta_segmentation->GetBufferPointer(), meta_segment_,
           segment_size * sizeof(unsigned char));
}

/******************************************************************************
 * set_meta_segmentation: constructs and sets meta_segmentation member
 * Arguments:
 *      meta_segment_: buffer containing lungs segmentation
 *      len: lenght of buffer
 *      v: segment label value
 *****************************************************************************/
void LiTS_scan::set_meta_segmentation(bool *meta_segment_, unsigned len,
                                      unsigned char v)
{
    meta_segmentation = SegmentationType::New();

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

    meta_segmentation->SetRegions(desiredRegion);
    meta_segmentation->Allocate();

    spacing[axes_order[0]] = voxel_w;
    spacing[axes_order[1]] = voxel_h;
    spacing[axes_order[2]] = voxel_d;
    meta_segmentation->SetSpacing(spacing);

    unsigned int segment_size = w * h * d;
    unsigned char *meta_segment_char = new unsigned char[segment_size];

    if (meta_segmentation.IsNotNull())
        memcpy(meta_segment_char, meta_segmentation->GetBufferPointer(),
               segment_size * sizeof(unsigned char));
    for(unsigned int i = 0; i < segment_size; i++)
    {
        if(meta_segment_[i])
            meta_segment_char[i] = v;
    }
    memcpy(meta_segmentation->GetBufferPointer(), meta_segment_,
           segment_size * sizeof(unsigned char));
    delete [] meta_segment_char;
}

/******************************************************************************
 * get_volume: returns volume pointer
 *****************************************************************************/
VolumeType::Pointer LiTS_scan::get_volume()
{
    return volume;
}

/******************************************************************************
 * get_segmentation: returns segmentation pointer
 *****************************************************************************/
SegmentationType::Pointer LiTS_scan::get_segmentation()
{
    return segmentation;
}

/******************************************************************************
 * get_meta_segmentation: returns meta segmentation pointer
 *****************************************************************************/
SegmentationType::Pointer LiTS_scan::get_meta_segmentation()
{
    return meta_segmentation;
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
 * save_meta_segmentation: save meta_segmentation at the input path
 * Arguments:
 *      meta_segmentation_path_: path where to save meta segmentation
 *****************************************************************************/
void LiTS_scan::save_meta_segmentation(std::string meta_segmentation_path_)
{
    SegmentationWriterType::Pointer segmentation_writer =
            SegmentationWriterType::New();

    segmentation_writer->SetFileName(meta_segmentation_path_);
    segmentation_writer->SetInput(meta_segmentation);
    segmentation_writer->Update();
}
