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

void liver_side_estimator_train_and_valid(LiTS_db db)
{
    unsigned N_iters = 10;
    unsigned N_subj_batch = 5;
    unsigned N_augment = 10;
    LiTS_liver_side_estimator liver_side_estimator;
    LiTS_processor p(-100, 400,-0.5, 0.5);
    float t_acc = 0;
    float v_acc = 0;
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    for(unsigned int it = 0; it < N_iters; it++)
    {
        t_acc = liver_side_estimator.train_liver_side_estimator(db, p,
                                                                N_iters,
                                                                N_subj_batch,
                                                                N_augment);
        if(it % 100 == 0)
            v_acc = liver_side_estimator.valid_liver_side_estimator(db, p);

        std::cout<<"(Iteration: "<<(it + 1)<<"/"<<N_iters<<"), (valid/train):";
        std::cout<<" ("<<v_acc<<"/"<<t_acc<<")\r"<<std::flush;
    }
    gettimeofday(&end_time, NULL);
    float runtime = (end_time.tv_sec-start_time.tv_sec) +
                    (end_time.tv_usec-start_time.tv_usec) * 1e-6;

    std::cout<<std::endl;
    std::cout<<"Time elapsed: "<<runtime<<std::endl;

}

