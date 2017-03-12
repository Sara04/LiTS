#include <iostream>
#include <stdlib.h>
#include <string.h>

#include <boost/filesystem.hpp>

#include "lits_db/lits_database.h"
#include "lits_scan/lits_scan.h"
#include "preprocessor/lits_preprocessor.h"
#include "visualize/lits_visualize.h"
#include "detectors/lung_detector.h"
#include "detectors/liver_detector.h"

#include <sys/time.h>

#include "itkExtractImageFilter.h"
#include "itkImageFileWriter.h"

typedef itk::ExtractImageFilter<VolumeType, VolumeType > ExtractVolumeType;
typedef itk::ExtractImageFilter<SegmentationType, SegmentationType > ExtractSegmentationType;
typedef itk::ImageFileWriter<VolumeType> VolumeWriterType;
typedef itk::ImageFileWriter<SegmentationType> SegmentationWriterType;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Invalid number of input arguments." << std::endl;
        std::cout << "First argument: directory path where the Train Batch 1";
        std::cout <<" and Train Batch 2 are placed."<< std::endl;
        exit(EXIT_FAILURE);
    }

    if (!strcmp(argv[1], "-h") or !strcmp(argv[1], "-help")
        or !strcmp(argv[1], "help") or !strcmp(argv[1], "h"))
    {
        std::cout << "comparison:" << strcmp(argv[1], "-h") << std::endl;
        std::cout
                << "First argument: directory path where the Train Batch 1 and Train Batch 2 are placed."
                << std::endl;
        exit(EXIT_SUCCESS);
    }

    boost::filesystem::path db_path(argv[1]);

    if (boost::filesystem::is_directory(db_path))
    {
        std::string db_path_str(argv[1]);
        boost::filesystem::path db_path_batch(
                (db_path_str + "/Training Batch 1").c_str());
        if (!boost::filesystem::is_directory(db_path_batch))
        {
            std::cout << "Training Batch 1 does not exists." << std::endl;
            exit(EXIT_FAILURE);
        }

        db_path_batch = db_path_str + "/Training Batch 2";
        if (!boost::filesystem::is_directory(db_path_batch))
        {
            std::cout << "Training Batch 2 does not exists." << std::endl;
            exit(EXIT_FAILURE);
        }

    }
    else
    {
        std::cout << "Invalid database path!" << std::endl;
        exit(EXIT_FAILURE);
    }

    LiTS_db db(db_path.string());
    db.load_train_subject_names();
    db.load_test_subject_names();
    db.train_data_split(0, 1);

    LiTS_preprocessor p;
    LiTS_lung_detector lung_d;
    LiTS_liver_detector liver_d;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    for (unsigned int i = 0; i < db.get_number_of_training(); i++)
    {
        std::string volume_path_in;
        std::string segmentation_path_in;
        std::string volume_path_out;
        std::string segmentation_path_out;
        std::string scan_name;
        scan_name = db.get_train_subject_name(i);

        std::cout<<"\n"<<std::endl;
        std::cout<<"scan name:"<<i<<" "<<scan_name<<std::endl;

        db.get_train_paths(scan_name, volume_path_in, segmentation_path_in);
        db.get_augmentation_paths(scan_name, volume_path_out, segmentation_path_out);

        LiTS_scan ls(volume_path_in, segmentation_path_in);

        ls.load_volume();
        ls.load_segmentation();
        ls.load_info();

        p.preprocess(&ls);
        lung_d.lung_segmentation(&ls);
        liver_d.estimate_liver_bounding_box(&ls);

        VolumeType::IndexType desired_start;
        VolumeType::SizeType desired_size;

        desired_start[0] = ls.get_liver_bbox()[0];
        desired_start[1] = ls.get_liver_bbox()[2];
        desired_start[2] = ls.get_liver_bbox()[4];

        desired_size[0] = ls.get_liver_bbox()[1] - ls.get_liver_bbox()[0];
        desired_size[1] = ls.get_liver_bbox()[3] - ls.get_liver_bbox()[2];
        desired_size[2] = ls.get_liver_bbox()[5] - ls.get_liver_bbox()[4];

        VolumeType::RegionType desiredRegion(desired_start, desired_size);

        ExtractVolumeType::Pointer extract_volume = ExtractVolumeType::New();
        extract_volume->SetExtractionRegion(desiredRegion);
        extract_volume->SetInput(ls.get_volume());

        ExtractSegmentationType::Pointer extract_segmentation = ExtractSegmentationType::New();
        extract_segmentation->SetExtractionRegion(desiredRegion);
        extract_segmentation->SetInput(ls.get_segmentation());

        VolumeWriterType::Pointer volume_writer = VolumeWriterType::New();
        volume_writer->SetFileName(volume_path_out);
        volume_writer->SetInput(extract_volume->GetOutput());
        volume_writer->Update();

        SegmentationWriterType::Pointer segmentation_writer = SegmentationWriterType::New();
        segmentation_writer->SetFileName(segmentation_path_out);
        segmentation_writer->SetInput(extract_segmentation->GetOutput());
        segmentation_writer->Update();

    }
    gettimeofday(&end_time, NULL);
    float runtime = (end_time.tv_sec - start_time.tv_sec)
    + (end_time.tv_usec - start_time.tv_usec) * 1e-6;

    std::cout << "Time elapsed:" << runtime << std::endl;

    return 0;
}
