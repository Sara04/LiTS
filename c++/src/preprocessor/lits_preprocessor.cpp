#include "preprocessor_cuda.cuh"
#include "lits_preprocessor.h"

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
LiTS_preprocessor::LiTS_preprocessor(float lt, float ut, float min, float max,
                                     std::string app)
{
    lower_threshold = lt;
    upper_threshold = ut;
    minimum_value = min;
    maximum_value = max;
    approach = app;
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
void LiTS_preprocessor::preprocess(LiTS_scan *scan)
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
        short desired_orientation[3] = {0, 1, 1};
        bool reorient = false;
        bool permute = false;
        for(unsigned int i = 1; i < 3; i++)
        {
            if(orientation[i] != desired_orientation[i])
                reorient = true;
            if(order[i] != i)
                permute = true;
        }

        if (reorient or permute)
        {
            VolumeType::DirectionType direction = scan->get_volume()->GetDirection();
            VolumeType::DirectionType desired_direction;
            desired_direction[0][0] = orientation[0];
            desired_direction[1][1] = 1;
            desired_direction[2][2] = 1;

            OrientVolumeType::Pointer orienter_v = OrientVolumeType::New();

            orienter_v->SetGivenCoordinateDirection(direction);
            orienter_v->SetInput(scan->get_volume());
            orienter_v->SetDesiredCoordinateDirection(desired_direction);
            orienter_v->UpdateLargestPossibleRegion();
            scan->set_volume(orienter_v->GetOutput());

            if((scan->get_segmentation()).IsNotNull())
            {
                OrientSegmentationType::Pointer orienter_s =
                        OrientSegmentationType::New();
                orienter_s->SetGivenCoordinateDirection(direction);
                orienter_s->SetInput(scan->get_segmentation());
                orienter_s->SetDesiredCoordinateDirection(desired_direction);
                orienter_s->UpdateLargestPossibleRegion();
                scan->set_segmentation(orienter_s->GetOutput());
            }
        }
    }
    else
    {

        unsigned char *segmentation;
        if(scan->get_segmentation().IsNotNull())
            segmentation = (scan->get_segmentation())->GetBufferPointer();
        else
            segmentation = NULL;
        preprocess_cuda((scan->get_volume())->GetBufferPointer(), segmentation,
                        scan->get_height(), scan->get_width(), scan->get_depth(),
                        order, orientation,
                        lower_threshold, upper_threshold,
                        minimum_value, maximum_value);

    }
}
