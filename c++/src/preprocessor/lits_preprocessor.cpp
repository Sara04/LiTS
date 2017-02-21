#include "preprocessor_cuda.h"
#include "lits_preprocessor.h"

LiTS_preprocessor::LiTS_preprocessor(float lt/*=-300*/, float ut/*=700.0*/,
		                             float min/*=-0.5*/, float max/*=0.5*/,
		                             std::string app/*='cuda'*/)
{
	lower_threshold = lt;
	upper_threshold = ut;
	minimum_value = min;
	maximum_value = max;
	approach=app;
}

LiTS_preprocessor::~LiTS_preprocessor()
{
	lower_threshold = NULL;
	upper_threshold = NULL;
	minimum_value = NULL;
	maximum_value = NULL;
	approach="";
}

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
			OrientVolumeType::Pointer orienter = OrientVolumeType::New();
			orienter->SetGivenCoordinateDirection(direction);
			orienter->SetInput(scan->get_volume());
			direction[1][1] = 1;
			orienter->SetDesiredCoordinateDirection(direction);
			orienter->UpdateLargestPossibleRegion();
			scan->set_volume(orienter->GetOutput());
		}
	}
	else
	{
		VolumeType::SizeType size = ((scan->get_volume())->GetLargestPossibleRegion()).GetSize();
		preprocess_cuda((scan->get_volume())->GetBufferPointer(),
				        size[0], size[1], size[2],
				        direction[1][1] < 0,
				        lower_threshold, upper_threshold, minimum_value, maximum_value);
	}
}
