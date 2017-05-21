/*
 * liver_side_estimator.h
 *
 *  Created on: Apr 15, 2017
 *      Author: sara
 */

#ifndef LIVER_SIDE_ESTIMATOR_H_
#define LIVER_SIDE_ESTIMATOR_H_

#include "../lits_db/lits_database.h"
#include "../lits_scan/lits_scan.h"
#include "../lits_processing/lits_processor.h"
#include "../neural_networks/nn.h"
#include <boost/progress.hpp>
#include <iostream>

/******************************************************************************
/* LiTS_liver_side_estimator a  class for the estimation at which side liver
 * is placed. It is necessary due to unreliable information about scan
 * orientation provided scans' headers.
 *
 * It has means of training and testing batch creation, estimation training and
 * testing.
 *
 * Attributes:
 *
 *      nn_clf: neural network based liver side classifier
 *      w_rs: width of the input images
 *      h_rs: height of the input images
 *      ext_d: how many mm below lungs bottom training/testing slices should
 *          be selected
 *      ext_u: how many mm above lungs bottom training/testing slices should
 *          be selected
 *      N_slices: number of selected slices
 *      nn_clf_on_gpu: flag indicating whether nn_clf is already transfered
 *          to the gpu
 *
 *      training_data: pointer to the training data
 *      training_labels: pointer to the training labels
 *      training_errors: pointer to the training errors
 *      testing_data: pointer to the testing data
 *      testing_labels: pointer to the testing labels
 *      testing_errors: pointer to the testing errors
 *
 * Methods:
 *
 *      LiTS_liver_side_estimator: constructor
 *      ~LiTS_liver_side_estimator: destructor
 *
 *      create_training_data: create training data
 *      create_testing_data: create testing data
 *
 *      train_liver_side_estimator: train neural network nn_clf
 *      test_liver_side_estimator: test neural network nn_clf
 *
 *****************************************************************************/
class LiTS_liver_side_estimator
{

private:

    NN nn_clf;
    unsigned w_rs;
    unsigned h_rs;
    float ext_d;
    float ext_u;
    unsigned N_slices;

    float *training_data;
    float *training_gt;
    float *training_errors;

    float *testing_data;
    float *testing_gt;
    float *testing_errors;

    bool nn_clf_on_gpu;

public:

    LiTS_liver_side_estimator(unsigned w_rs_=64, unsigned h_rs_=48,
                              float ext_d_=25.0, float ext_u_=5.0);
    ~LiTS_liver_side_estimator();

    void create_training_data(std::vector<LiTS_scan> train_scans_batch,
                              unsigned N_augment);

    void create_testing_data(std::vector<LiTS_scan> test_scans_batch);

    void train_liver_side_estimator(LiTS_db &db, LiTS_processor &p,
                                    unsigned N_iters,
                                    unsigned N_subj_batch,
                                    unsigned N_augment);

    void test_liver_side_estimator();
};

#endif /* LIVER_SIDE_ESTIMATOR_H_ */
