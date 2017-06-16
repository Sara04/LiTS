/*
 * liver_side_estimator.cpp
 *
 *  Created on: May 21, 2017
 *      Author: sara
 */

#include "liver_side_estimation.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/******************************************************************************
 * save_liver_side_ground_truth: function of liver side ground truth
 * calculation based on liver and lungs segmentation masks
 *
 * arguments:
 * 		db: LiTS_db object
 *****************************************************************************/
void save_liver_side_ground_truth(LiTS_db db)
{
    LiTS_liver_side_estimator liver_side_estimator;
    LiTS_processor p(-200, 500,-0.5, 0.5);
    boost::progress_display show_progress( db.get_number_of_training() );
    for(unsigned i = 0; i < db.get_number_of_training() ; i++)
    {
        std::string scan_name = db.get_train_subject_name(i);
        std::string volume_path, segment_path;
        std::string meta_segment_path, liver_gt_path;
        db.get_train_paths(scan_name, volume_path, segment_path);
        db.get_train_meta_segment_path(scan_name, meta_segment_path);
        db.get_train_liver_side_gt_path(scan_name, liver_gt_path);
        LiTS_scan ls(volume_path, segment_path, meta_segment_path);
        liver_side_estimator.load_and_reorient_masks(p, ls);

        unsigned char ** mask_gt = new unsigned char*[1];
        unsigned char ** mask_m = new unsigned char*[1];
        unsigned S[3];
        float vox_S[3];
        unsigned int *Ls = new unsigned int[2];

        liver_side_estimator.get_volume_and_voxel_sizes(ls, S, vox_S);
        Ls[0] = 0;
        Ls[1] = S[0] * S[1] * S[2];
        mask_m[0] = (ls.get_meta_segment())->GetBufferPointer();
        unsigned int *B = new unsigned int[6];
        extract_lung_bounds(mask_m, S, Ls, 1, B);
        bool *gt = new bool[1];
        mask_gt[0] = (ls.get_segment())->GetBufferPointer();
        extract_liver_side_ground_truth(mask_gt, S, Ls, 1, B, gt);

        std::ofstream gt_file;
        gt_file.open(liver_gt_path);
        gt_file<<gt[0]<<std::endl;
        gt_file.close();
        delete [] mask_gt;
        delete [] mask_m;
        delete [] Ls;
        delete [] gt;
        delete [] B;
        ++show_progress;
    }
}

/******************************************************************************
 * liver_side_estimator_train_and_valid: this function runs liver side
 * estimation model training, validation and saving. Since the information
 * about orientation of the axis that is normal to the sagittal plane is not
 * reliable and it is considered to be important for the further liver and
 * tumor segmentation, a model is trained to estimate its orientation
 * (or liver's side)
 *
 * arguments:
 * 		model_path: path where to store model
 * 		db: LiTS_db object
 * 		N_iters: number of iterations
 * 		N_subj_t: number of subjects for training
 * 		N_subj_v: number of subjects for validation
 * 		N_augment_t: augmentation factor of training data
 * 		lr: learning rate
 *****************************************************************************/
void liver_side_estimator_train_and_valid(std::string model_path, LiTS_db db,
                                          unsigned N_iters,
                                          unsigned N_subj_t,
                                          unsigned N_subj_v,
                                          unsigned N_augment_t,
                                          float lr)
{
    LiTS_liver_side_estimator liver_side_estimator(model_path);
    LiTS_processor p(-100, 400,-0.5, 0.5);
    float t_acc = 0;
    float v_acc = 0;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    for(unsigned int it = 0; it < N_iters; it++)
    {
        if(it == 1000)
            lr *= 0.1;
        t_acc = liver_side_estimator.develop_liver_side_estimator(db, p,
                                                                  N_subj_t,
                                                                  N_augment_t,
                                                                  lr, true);
        if(it % 20 == 0)
            v_acc = liver_side_estimator.valid_liver_side_estimator(db, p,
                                                                    N_subj_v,
                                                                    true);

        std::cout<<"(Iteration: "<<(it + 1)<<"/"<<N_iters<<"), (valid/train):";
        std::cout<<" ("<<v_acc<<"/"<<t_acc<<")\r"<<std::flush;

        if((it % 100) == 0)
            liver_side_estimator.save_model(it);
    }

    gettimeofday(&end_time, NULL);
    float runtime = (end_time.tv_sec-start_time.tv_sec) +
                    (end_time.tv_usec-start_time.tv_usec) * 1e-6;

    std::cout<<std::endl;
    std::cout<<"Time elapsed for liver side estimator model is: ";
    std::cout<<runtime<<std::endl;

}

void liver_side_estimator_eval(std::string model_path, LiTS_db db,
                               std::string scan_name)
{
    LiTS_liver_side_estimator liver_side_estimator(model_path);
    LiTS_processor p(-200, 500,-0.5, 0.5);

    liver_side_estimator.eval_liver_side_estimator(db, p, scan_name, true);
}

void estimate_liver_side(std::string model_path, LiTS_db db,
                         std::string scan_name)
{
    LiTS_liver_side_estimator liver_side_estimator(model_path);
    LiTS_processor p(-200, 500,-0.5, 0.5);

    liver_side_estimator.estimate_liver_side(db, p, scan_name, true);
}
