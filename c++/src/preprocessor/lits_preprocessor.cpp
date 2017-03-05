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
 * if necessary flip along front-back body direction so
 * that body is oriented in RAS coordinate system
 *
 * ?????????????????????????????????????????????????????????
 *	Should implement body orientation estimator since
 *	coordinate systems provided in headers are not reliable
 *	here it is assumed that information about orientation
 *	along top-bottom and front-back body axes is correct
 * ?????????????????????????????????????????????????????????
 *
 * Arguments:
 * 		scan: pointer to the LiTS_scan object
 */
void LiTS_preprocessor::preprocess(LiTS_scan *scan)
{
    VolumeType::DirectionType direction = scan->get_volume()->GetDirection();

    if (!strcmp(approach.c_str(), "itk"))
    {
        RescalerType::Pointer rescaler = RescalerType::New();
        rescaler->SetInput(scan->get_volume());
        rescaler->SetWindowMinimum(lower_threshold);
        rescaler->SetWindowMaximum(upper_threshold);
        rescaler->SetOutputMinimum(minimum_value);
        rescaler->SetOutputMaximum(maximum_value);
        rescaler->UpdateLargestPossibleRegion();

        scan->set_volume(rescaler->GetOutput());

        if (direction[1][1] < 0)
        {
            OrientVolumeType::Pointer orienter_v = OrientVolumeType::New();
            OrientSegmentationType::Pointer orienter_s =\

                    OrientSegmentationType::New();

            orienter_v->SetGivenCoordinateDirection(direction);
            orienter_v->SetInput(scan->get_volume());
            orienter_s->SetGivenCoordinateDirection(direction);
            orienter_s->SetInput(scan->get_segmentation());
            direction[1][1] = 1;
            orienter_v->SetDesiredCoordinateDirection(direction);
            orienter_v->UpdateLargestPossibleRegion();
            orienter_s->SetDesiredCoordinateDirection(direction);
            orienter_s->UpdateLargestPossibleRegion();

            scan->set_volume(orienter_v->GetOutput());
            scan->set_segmentation(orienter_s->GetOutput());
        }
    }
    else
    {
        preprocess_cuda((scan->get_volume())->GetBufferPointer(),
                        (scan->get_segmentation())->GetBufferPointer(),
                        scan->get_height(), scan->get_width(),
                        scan->get_depth(), direction[1][1] < 0, lower_threshold,
                        upper_threshold, minimum_value, maximum_value);
    }
}
