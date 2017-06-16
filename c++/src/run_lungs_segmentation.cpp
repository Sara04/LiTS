/*
 * run_lungs_segmentation.cpp
 *
 *  Created on: May 21, 2017
 *      Author: sara
 */

/******************************************************************************
 * This functions runs lungs segmentation of all training examples.
 * It is based on the voxel value thresholding, region growing and mathematical
 * morphology operations. Before the segmentation, voxel values are clipped and
 * normalized, and volume's axes are re-ordered and re-oriented to RAS or LAS
 * coordinate system.
 * !!! Note that the information provided in the scan's header about the axis
 * orientation that is normal to the sagittal plane is not reliable!!!
 * After the segmentation, axes of the meta-segmentation are re-ordered and
 * re-oriented to the original state.
 *****************************************************************************/
#include "run_lungs_segmentation.h"

void run_train_lungs_segmentation(LiTS_db db)
{
    LiTS_processor p;
    LiTS_lung_segmentator lung_s;

    boost::progress_display show_progress( db.get_number_of_training() );
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    for(unsigned int i = 0; i < db.get_number_of_training(); i++)
    {
        std::string volume_path;
        std::string meta_segment_path;
        std::string scan_name;

        scan_name = db.get_train_subject_name(i);
        db.get_train_volume_path(scan_name, volume_path);
        db.get_train_meta_segment_path(scan_name, meta_segment_path);

        LiTS_scan ls(volume_path);
        ls.load_volume();
        ls.load_info();

        p.preprocess_volume(&ls);
        lung_s.lung_segmentation(&ls);

        p.reorient_segment(ls.get_meta_segment()->GetBufferPointer(),
                           ls.get_width(), ls.get_height(), ls.get_depth(),
                           p.get_axes_order(), p.get_axes_orient(),
                           ls.get_axes_order(), ls.get_axes_orient());

        ls.save_meta_segment(meta_segment_path);

        ++show_progress;
    }
    gettimeofday(&end_time, NULL);
    float runtime = (end_time.tv_sec-start_time.tv_sec) +
                    (end_time.tv_usec-start_time.tv_usec) * 1e-6;

    std::cout<<"Time elapsed:"<< runtime<<std::endl;
}



