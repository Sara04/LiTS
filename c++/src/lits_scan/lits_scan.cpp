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
LiTS_scan::LiTS_scan(std::string volume_path_, std::string segment_path_)
{
    volume = VolumeType::New();
    segment = SegmentType::New();

    volume_path = volume_path_;
    segment_path = segment_path_;

    for(unsigned int i = 0; i < 3; i++)
    {
        axes_order[i] = i;
        axes_orient[i] = 1;
    }
}

/******************************************************************************
 * LiTS_scan constructor: assigning volume, segmentation and meta segmentation
 * file paths, volume and segmentation member dynamic construction,
 *  initializing axes_order and axes_orientation members
 *
 * Arguments:
 *      volume_path_: path to the volume file
 *      segment_path: path to the segmentation (ground truth) file
 *      meta_segment_path: path to the meta segmentation file
 *****************************************************************************/
LiTS_scan::LiTS_scan(std::string volume_path_, std::string segment_path_,
                     std::string meta_segment_path_)
{
    volume = VolumeType::New();
    segment = SegmentType::New();
    meta_segment = SegmentType::New();

    volume_path = volume_path_;
    segment_path = segment_path_;
    meta_segment_path = meta_segment_path_;

    for(unsigned int i = 0; i < 3; i++)
    {
        axes_order[i] = i;
        axes_orient[i] = 1;
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
        axes_orient[i] = 1;
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
 * load_segment: loading segmentation from the segmentation_path using
 * segmentation_reader
 *****************************************************************************/
void LiTS_scan::load_segment()
{
    if (segment.IsNotNull())
    {
        SegmentReaderType::Pointer segment_reader = SegmentReaderType::New();
        segment_reader->SetFileName(segment_path);
        segment_reader->Update();
        segment = segment_reader->GetOutput();
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
        axes_orient[axes_order[i]] = direction_v[i][axes_order[i]];
    }

    w = size_v[axes_order[0]];
    h = size_v[axes_order[1]];
    d = size_v[axes_order[2]];

    voxel_w = spacing[axes_order[0]];
    voxel_h = spacing[axes_order[1]];
    voxel_d = spacing[axes_order[2]];

    if (segment.IsNotNull())
    {
        SegmentType::RegionType s_region = segment->GetLargestPossibleRegion();
        SegmentType::SizeType size_s = s_region.GetSize();

        if (size_s[0])
        {
            if (size_v[0] != size_s[0] or size_v[1] != size_s[1]
                or size_v[2] != size_s[2])
            {
                std::cout << "Volume path:" << volume_path << std::endl;
                std::cout << "Segmentation path:" << segment_path << std::endl;
                std::cout << "Volume and segmentation data are not compatible";
                std::cout << "\n";
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
 * load_meta_segment: loading meta segmentation from the
 * meta_segmentation_path using segmentation_reader
 *****************************************************************************/
void LiTS_scan::load_meta_segment()
{

    if (fs::exists(meta_segment_path))
    {
        SegmentReaderType::Pointer segment_reader = SegmentReaderType::New();
        meta_segment = SegmentType::New();
        segment_reader->SetFileName(meta_segment_path);
        segment_reader->Update();
        meta_segment = segment_reader->GetOutput();
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
 * set_segment: sets segmentation member
 * Arguments:
 *      segment_: pointer to the segmentation data
 *****************************************************************************/
void LiTS_scan::set_segment(SegmentType::Pointer segment_)
{
    segment = segment_;
}

/******************************************************************************
 * set_meta_segment: sets meta_segment member
 * Arguments:
 *      meta_seg: pointer to the meta segmentation data
 *****************************************************************************/
void LiTS_scan::set_meta_segment(SegmentType::Pointer meta_seg)
{
    meta_segment = meta_seg;
}

/******************************************************************************
 * set_meta_segment: constructs and sets meta_segmentation member
 * Arguments:
 *      meta_seg: buffer containing lungs segmentation
 *****************************************************************************/
void LiTS_scan::set_meta_segment(unsigned char *meta_seg)
{
    meta_segment = SegmentType::New();

    SegmentType::IndexType desired_start;
    SegmentType::SizeType desired_size;
    SegmentType::SpacingType spacing;

    for(unsigned int i = 0;i < 3; i++)
        desired_start[i] = 0;

    desired_size[axes_order[0]] = w;
    desired_size[axes_order[1]] = h;
    desired_size[axes_order[2]] = d;

    SegmentType::RegionType desiredRegion(desired_start, desired_size);

    meta_segment->SetRegions(desiredRegion);
    meta_segment->Allocate();

    spacing[axes_order[0]] = voxel_w;
    spacing[axes_order[1]] = voxel_h;
    spacing[axes_order[2]] = voxel_d;

    meta_segment->SetSpacing(spacing);

    memcpy(meta_segment->GetBufferPointer(), meta_seg,
           w * h * d * sizeof(unsigned char));
}

/******************************************************************************
 * set_meta_segment: constructs and sets meta_segmentation member
 * it is assumed that meta segmentations have the same orientation
 * Arguments:
 *      meta_seg: buffer containing lungs segmentation
 *      len: lenght of buffer
 *      v: segment label value
 *****************************************************************************/
void LiTS_scan::set_meta_segment(bool *meta_seg, unsigned len, unsigned char v)
{
    unsigned char *meta_seg_ch = new unsigned char[w * h * d];

    if (meta_segment.IsNotNull())
        memcpy(meta_seg_ch, meta_segment->GetBufferPointer(),
               w * h * d * sizeof(unsigned char));
    else
    {
        meta_segment = SegmentType::New();

        SegmentType::IndexType desired_start;
        SegmentType::SizeType desired_size;
        SegmentType::SpacingType spacing;

        for(unsigned int i = 0; i < 3; i++)
            desired_start[i] = 0;

        desired_size[axes_order[0]] = w;
        desired_size[axes_order[1]] = h;
        desired_size[axes_order[2]] = d;

        SegmentType::RegionType desiredRegion(desired_start, desired_size);

        meta_segment->SetRegions(desiredRegion);
        meta_segment->Allocate();

        spacing[axes_order[0]] = voxel_w;
        spacing[axes_order[1]] = voxel_h;
        spacing[axes_order[2]] = voxel_d;
        meta_segment->SetSpacing(spacing);
    }

    for(unsigned int i = 0; i < w * h * d; i++)
        if(meta_seg[i])
            meta_seg_ch[i] = v;

    memcpy(meta_segment->GetBufferPointer(), meta_seg_ch,
           w * h * d * sizeof(unsigned char));

    delete [] meta_seg_ch;
}

/******************************************************************************
 * get_volume: returns volume pointer
 *****************************************************************************/
VolumeType::Pointer LiTS_scan::get_volume()
{
    return volume;
}

/******************************************************************************
 * get_segment: returns segmentation pointer
 *****************************************************************************/
SegmentType::Pointer LiTS_scan::get_segment()
{
    return segment;
}

/******************************************************************************
 * get_meta_segment: returns meta segment pointer
 *****************************************************************************/
SegmentType::Pointer LiTS_scan::get_meta_segment()
{
    return meta_segment;
}

/******************************************************************************
 * get_axes_order: returns pointer to axes order
 *****************************************************************************/
short int * LiTS_scan::get_axes_orient()
{
    return axes_orient;
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
 * save_meta_segment: save meta_segmentation at the input path
 * Arguments:
 *      meta_segment_path_: path where to save meta segmentation
 *****************************************************************************/
void LiTS_scan::save_meta_segment(std::string meta_segment_path_)
{
    SegmentWriterType::Pointer segment_writer = SegmentWriterType::New();

    segment_writer->SetFileName(meta_segment_path_);
    segment_writer->SetInput(meta_segment);
    segment_writer->Update();
}
