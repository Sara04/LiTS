/*
 * liver_side_estimator.cpp
 *
 *  Created on: May 21, 2017
 *      Author: sara
 */
/******************************************************************************
 * This functions runs liver side estimation model training, validation and
 * saving. Since the information about orientation of the axis that is normal
 * to the sagittal plane is not reliable and it is considered to be important
 * for the further liver and tumor segmentation, a model is trained to estimate
 * its orientation (or liver's side).
 *****************************************************************************/

#include "liver_side_estimation.h"

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
        t_acc = liver_side_estimator.develop_liver_side_estimator(db, p,
                                                                  N_iters,
                                                                  N_subj_t,
                                                                  N_augment_t,
                                                                  lr,
                                                                  true);
        if(it % 100 == 0)
            v_acc = liver_side_estimator.valid_liver_side_estimator(db, p,
                                                                    N_subj_v,
                                                                    true);

        std::cout<<"(Iteration: "<<(it + 1)<<"/"<<N_iters<<"), (valid/train):";
        std::cout<<" ("<<v_acc<<"/"<<t_acc<<")\r"<<std::flush;
    }
    liver_side_estimator.save_model();
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
    LiTS_processor p(-100, 400,-0.5, 0.5);

    liver_side_estimator.eval_liver_side_estimator(db, p, scan_name, true);
}

void estimate_liver_side(std::string model_path, LiTS_db db,
                         std::string scan_name)
{
    LiTS_liver_side_estimator liver_side_estimator(model_path);
    LiTS_processor p(-100, 400,-0.5, 0.5);

    liver_side_estimator.estimate_liver_side(db, p, scan_name, true);
}
