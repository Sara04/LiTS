#include "pre_and_post_processor_cuda.cuh"
#include "lits_pre_and_post_processor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/*
 * LiTS_preprocessor constructor: assigning range thresholds and
 * new range minimum and maximum values and approach
 *
 * Arguments:
 * 		lt: lower threshold of original voxel intensity
 * 		ut: upper threshold of original voxel intensity
 * 		min: minimum value of voxel intensity in new intensity range
 * 		max: maximum value of voxel intensity in new intensity range
 * 		app: "itk"/"cuda" approach
 */
LiTS_pre_and_post_processor::
LiTS_pre_and_post_processor(float lt, float ut, float min, float max,
                            std::string app)
{
    lower_threshold = lt;
    upper_threshold = ut;
    minimum_value = min;
    maximum_value = max;
    approach = app;

    for(unsigned int i = 0; i < 3; i++)
    {
        ord[i] = i;
        orient[i] = 1;

    }
}

/*
 * preprocess: normalize voxel intensities and
 * order and/or flip axes if necessary
 * it is assumed that:
 * 1. order of the axes provided in header is correct
 * 2. orientations of the last two axes are correct
 *    (it is assumed that orientation label of the first
 *     axes left-right is not convenient, views used
 *     by radiologists and neurologists)
 *
 * ?????????????????????????????????????????????????????????
 *	Should implement classifier of the volume orientation
 *	along the first axi
 * ?????????????????????????????????????????????????????????
 *
 * Arguments:
 * 		scan: pointer to the LiTS_scan object
 */
void LiTS_pre_and_post_processor::preprocess_volume(LiTS_scan *scan)
{
    unsigned int *order = scan->get_axes_order();
    short *orientation = scan->get_axes_orientation();
    if (!strcmp(approach.c_str(), "itk"))
    {
        // 1. Voxel values range re-scaling
        RescalerType::Pointer rescaler = RescalerType::New();
        rescaler->SetInput(scan->get_volume());
        rescaler->SetWindowMinimum(lower_threshold);
        rescaler->SetWindowMaximum(upper_threshold);
        rescaler->SetOutputMinimum(minimum_value);
        rescaler->SetOutputMaximum(maximum_value);
        rescaler->UpdateLargestPossibleRegion();

        scan->set_volume(rescaler->GetOutput());

        // 2. Permute and re-orient axes if necessary
        bool reorient = false;
        bool permute = false;
        for(unsigned int i = 1; i < 3; i++)
        {
            if(orientation[i] != orient[i])
                reorient = true;
            if(order[i] != order[i])
                permute = true;
        }

        if (reorient or permute)
        {
            VolumeType::DirectionType direction = scan->get_volume()->GetDirection();
            VolumeType::DirectionType desired_orientation;
            desired_orientation[0][0] = orientation[0];
            desired_orientation[1][1] = orient[1];
            desired_orientation[2][2] = orient[2];

            OrientVolumeType::Pointer orienter_v = OrientVolumeType::New();

            orienter_v->SetGivenCoordinateDirection(direction);
            orienter_v->SetInput(scan->get_volume());
            orienter_v->SetDesiredCoordinateDirection(desired_orientation);
            orienter_v->UpdateLargestPossibleRegion();
            scan->set_volume(orienter_v->GetOutput());
        }
    }
    else
        preprocess_volume_cuda((scan->get_volume())->GetBufferPointer(),
                               scan->get_width(), scan->get_height(), scan->get_depth(),
                               order, orientation,
                               lower_threshold, upper_threshold,
                               minimum_value, maximum_value);
}

void LiTS_pre_and_post_processor::normalize_volume(LiTS_scan *scan)
{
    if (!strcmp(approach.c_str(), "itk"))
    {
        // 1. Voxel values range re-scaling
        RescalerType::Pointer rescaler = RescalerType::New();
        rescaler->SetInput(scan->get_volume());
        rescaler->SetWindowMinimum(lower_threshold);
        rescaler->SetWindowMaximum(upper_threshold);
        rescaler->SetOutputMinimum(minimum_value);
        rescaler->SetOutputMaximum(maximum_value);
        rescaler->UpdateLargestPossibleRegion();

        scan->set_volume(rescaler->GetOutput());
    }
    else
        normalize_volume_cuda((scan->get_volume())->GetBufferPointer(),
                              scan->get_width(), scan->get_height(), scan->get_depth(),
                              lower_threshold, upper_threshold,
                              minimum_value, maximum_value);
}


void LiTS_pre_and_post_processor::reorient_volume(LiTS_scan *scan,
                                                  unsigned *cord,
                                                  short *corient,
                                                  unsigned *dord,
                                                  short *dorient)
{
    if (!strcmp(approach.c_str(), "itk"))
    {

        // 1. Permute and re-orient axes if necessary
        bool reorient = false;
        bool permute = false;
        for(unsigned int i = 1; i < 3; i++)
        {
            if(corient[i] != dorient[i])
                reorient = true;
            if(cord[i] != dord[i])
                permute = true;
        }

        if (reorient or permute)
        {
            VolumeType::DirectionType direction = scan->get_volume()->GetDirection();
            VolumeType::DirectionType desired_orientation;
            desired_orientation[0][0] = corient[0];
            desired_orientation[1][1] = dorient[1];
            desired_orientation[2][2] = dorient[2];

            OrientVolumeType::Pointer orienter_v = OrientVolumeType::New();
            orienter_v->SetGivenCoordinateDirection(direction);
            orienter_v->SetInput(scan->get_volume());
            orienter_v->SetDesiredCoordinateDirection(desired_orientation);
            orienter_v->UpdateLargestPossibleRegion();
            scan->set_volume(orienter_v->GetOutput());
        }
    }
    else
        reorient_volume_cuda((scan->get_volume())->GetBufferPointer(),
                             scan->get_width(), scan->get_height(), scan->get_depth(),
                             cord, corient, dord, dorient);
}

void LiTS_pre_and_post_processor::reorient_segmentation(LiTS_scan *scan,
                                                        unsigned *cord,
                                                        short *corient,
                                                        unsigned *dord,
                                                        short *dorient)
{
    if (!strcmp(approach.c_str(), "itk"))
    {
        // 1. Permute and re-orient axes if necessary
        bool reorient = false;
        bool permute = false;
        for(unsigned int i = 1; i < 3; i++)
        {
            if(corient[i] != dorient[i])
                reorient = true;
            if(cord[i] != dord[i])
                permute = true;
        }

        if (reorient or permute)
        {
            SegmentationType::DirectionType direction = scan->get_segmentation()->GetDirection();
            SegmentationType::DirectionType desired_orientation;
            desired_orientation[0][0] = corient[0];
            desired_orientation[1][1] = dorient[1];
            desired_orientation[2][2] = dorient[2];

            OrientSegmentationType::Pointer orienter_s = OrientSegmentationType::New();
            orienter_s->SetGivenCoordinateDirection(direction);
            orienter_s->SetInput(scan->get_segmentation());
            orienter_s->SetDesiredCoordinateDirection(desired_orientation);
            orienter_s->UpdateLargestPossibleRegion();
            scan->set_segmentation(orienter_s->GetOutput());
        }
    }
    else
        reorient_segmentation_cuda((scan->get_segmentation())->GetBufferPointer(),
                                   scan->get_width(), scan->get_height(), scan->get_depth(),
                                   cord, corient, dord, dorient);
}


void LiTS_pre_and_post_processor::reorient_segmentation(unsigned char * segmentation,
                                                        unsigned w, unsigned h, unsigned d,
                                                        unsigned *cord,
                                                        short *corient,
                                                        unsigned *dord,
                                                        short *dorient)
{
    reorient_segmentation_cuda(segmentation,
                               w, h, d,
                               cord, corient,
                               dord, dorient);
}

short * LiTS_pre_and_post_processor::get_axes_orientation()
{
    return orient;
}

unsigned * LiTS_pre_and_post_processor::get_axes_order()
{
    return ord;
}
