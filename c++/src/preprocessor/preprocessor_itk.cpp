#include "preprocessor_itk.h"

LiTS_preprocessor::LiTS_preprocessor(float lt/*=-300*/, float ut/*=700.0*/,
		                             float min/*=-0.5*/, float max/*=0.5*/)
{
	lower_threshold = lt;
	upper_threshold = ut;
	minimum_value = min;
	maximum_value = max;
	rescaler = RescalerType::New();
}

LiTS_preprocessor::~LiTS_preprocessor()
{
	lower_threshold = NULL;
	upper_threshold = NULL;
	minimum_value = NULL;
	maximum_value = NULL;
}

void LiTS_preprocessor::preprocess(LiTS_scan *scan)
{

	rescaler->SetInput(scan->get_volume());
	rescaler->SetWindowMinimum(lower_threshold);
	rescaler->SetWindowMaximum(upper_threshold);
	rescaler->SetOutputMinimum(minimum_value);
	rescaler->SetOutputMaximum(maximum_value);
	rescaler->UpdateLargestPossibleRegion();

	scan->set_volume(rescaler->GetOutput());

	VolumeType::DirectionType direction = scan->get_volume()->GetDirection();

	if (direction[1][1] < 0)
	{
		OrientVolumeType::Pointer orienter = OrientVolumeType::New();
		orienter->SetGivenCoordinateDirection(direction);
		orienter->SetInput(scan->get_volume());
		direction[1][1] = 1;
		orienter->SetDesiredCoordinateDirection(direction);
		orienter->UpdateLargestPossibleRegion();
		scan->set_volume(orienter->GetOutput());
	}

}
