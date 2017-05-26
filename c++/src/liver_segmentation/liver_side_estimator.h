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
/* LiTS_liver_side_estimator a class for liver side estimation model creation
 * It has means of neural network model initialization, loading and
 * pre-processing volume, segmentation and meta segmentation data, liver
 * side estimation model development, validation, evaluation and testing
 *
 * Attributes:
 *
 *      model_path: path to the directory where the input data mean, std and
 *          liver side estimation model will be/are saved
 *      nn_clf: neural network based liver side estimation model
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
 *      mean: pointer to the development data mean
 *      std: pointer to the development data standard deviation
 *
 *      develop_data: pointer to the development (model training) data
 *      develop_gt: pointer to the development (model training) ground truths
 *      develop_errors: pointer to the development (model training) errors
 *
 *      validate_data: pointer to the model validation data
 *      validate_gt: pointer to the model validation ground truths
 *      validate_errors: pointer to the model validation errors
 *
 *      eval_data: pointer to the model evaluation data
 *      eval_gt: pointer to the model evaluation ground truths
 *      eval_errors: pointer to the model evaluation errors
 *
 *      test_data: pointer to the testing data (no ground truths available)
 *
 * Methods:
 *
 *      LiTS_liver_side_estimator: constructor
 *      ~LiTS_liver_side_estimator: destructor
 *
 *      compute_mean: compute mean of the development images
 *      compute_std: compute standard deviation of the development images
 *      save_mean: save mean array in LSE model's directory
 *      save_std: save std in LSE model's directory
 *      load_mean: load mean array from LSE model's directory
 *      load_std: load std array from LSE model's directory
 *      create_input_data: create input data for model development, validation,
 *      evaluation or testing
 *
 *      develop_liver_side_estimator: train neural network nn_clf
 *      valid_liver_side_estimator: validate neural network nn_clf
 *      eval_liver_side_estimator: evaluate neural network nn_clf
 *      estimate_liver_side: estimate liver side
 *
 *****************************************************************************/
class LiTS_liver_side_estimator
{

private:

    std::string model_path;
    NN nn_clf;
    unsigned w_rs;
    unsigned h_rs;
    float ext_d;
    float ext_u;
    unsigned N_slices;

    float *mean;
    float *std;

    float *develop_data;
    float *develop_gt;
    float *develop_errors;

    float *validate_data;
    float *validate_gt;
    float *validate_errors;

    float *eval_data;
    float *eval_gt;
    float *eval_errors;

    float *test_data;

    bool nn_clf_on_gpu;

public:

    LiTS_liver_side_estimator(std::string model_path_,
                              unsigned w_rs_=64, unsigned h_rs_=48,
                              float ext_d_=25.0, float ext_u_=5.0);
    ~LiTS_liver_side_estimator();

    void compute_mean(LiTS_db &db, LiTS_processor &p);
    void compute_std(LiTS_db &db, LiTS_processor &p);

    void save_mean();
    void save_std();
    void save_model();

    void load_mean();
    void load_std();
    void load_model();

    void load_and_preprocess_scan(LiTS_processor &p, LiTS_scan &ls);

    void get_volume_and_voxel_sizes(LiTS_scan &ls, unsigned *S, float *vox_S);

    void create_input_data(std::vector<LiTS_scan> scans, std::string mode,
                           unsigned N_augment);

    float develop_liver_side_estimator(LiTS_db &db,
                                       LiTS_processor &p,
                                       unsigned N_iters,
                                       unsigned N_subj_batch,
                                       unsigned N_augment,
                                       float learning_rate,
                                       bool normalize=false);

    float valid_liver_side_estimator(LiTS_db &db,
                                     LiTS_processor &p,
                                     unsigned N_subj_batch,
                                     bool normalize=false);

    float eval_liver_side_estimator(LiTS_db &db,
                                    LiTS_processor &p,
                                    std::string scan_name,
                                    bool normalize=false);

    float estimate_liver_side(LiTS_db &db,
                              LiTS_processor &p,
                              std::string scan_name,
                              bool normalize=false);

};

#endif /* LIVER_SIDE_ESTIMATOR_H_ */
