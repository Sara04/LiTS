/*
 * lits_database.h
 *
 *  Created on: Feb 11, 2017
 *      Author: sara
 */

#ifndef LITS_DATABASE_H_
#define LITS_DATABASE_H_

#include <string>
#include <vector>
#include <iostream>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <cstdlib>

namespace fs = boost::filesystem;

/*
 * Class LiTS_db is a class for database management.
 *
 * It has means of creating lists of all available subjects
 * in the database from training and testing batches;
 * splitting the training data into development, validation
 * and evaluation parts; getting the number and subjects of the
 * training, development, validation, evaluation and testing
 * subsets;
 * getting the volume and segmentation paths of the training
 * and testing scans
 *
 * Attributes:
 *
 * 		db_path: path to the directory which contains folders
 * 		    "Training Batch 1", "Training Batch 2" and
 * 		    "Testing Batch"
 *
 * 		training_subjects: vector for storing subjects' names
 * 			from the training batches
 * 		development_subjects: vector for storing subjects' names
 * 			for algorithm development/training
 * 		validation_subjects: vector for storing subjects' names
 * 			for algorithm validation
 * 		evaluation_subjects: vector for storing subjects' names
 * 			for algorithm evaluation
 * 		testing_subjects: vector for storing subjects' names
 * 			from the testing batch
 *
 * 		n_train: the total number of training subjects
 * 		n_develop: the total number of development subjects
 * 		n_valid: the total number of validation subjects
 * 		n_eval: the total number of evaluation subjects
 * 		n_test: the total number of testing subjects
 *
 * Methods:
 *
 * 		LiTS_db: constructor
 *
 * 		load_train_subjects_names: loading train subjects' names
 * 		load_test_subjects_names: loading test subjects' names
 *
 * 		train_data_split: splitting data into development, validation
 * 			and evaluation parts
 *
 * 		empty_split: reseting data split
 *
 * 		get_number_of_training: get the total number of training subjects
 * 		get_number_of_development: get the total number of development subjects
 * 		get_number_of_validation: get the total number of validation subjects
 * 		get_number_of_evaluation: get the total number of evaluation subjects
 * 		get_number_of_testing: get the total number of testing subjects
 *
 * 		get_train_subject_name: get subject's name from the training set
 * 			at required position
 * 		get_develop_subject_name: get subject's name from the development set
 * 			at required position
 * 		get_valid_subject_name: get subject's name from the validation set
 * 			at required position
 * 		get_eval_subject_name: get subject's name from the evaluation set
 * 			at required position
 * 		get_test_subject_name: get subject's name from the testing set
 * 			at required position
 *
 * 		get_train_paths: get training subject's volume and segmentation paths
 * 		get_train_volume_path: get training subject's volume path
 * 		get_train_segmentation_path: get training subject's segmentation path
 * 		get_train_meta_segmentation_path: get training subject's
 * 		    meta segmentation path
 *      get_test_volume_path: get testing subject's volume path
 *      get_test_segmentation_path: get testing subject's segmentation path
 *
 */

class LiTS_db
{

private:

    std::string db_path;
    std::vector<std::string> training_subjects;
    std::vector<std::string> development_subjects;
    std::vector<std::string> validation_subjects;
    std::vector<std::string> evaluation_subjects;
    std::vector<std::string> testing_subjects;

    int n_train;
    int n_develop;
    int n_valid;
    int n_eval;
    int n_test;

public:

    LiTS_db(std::string db_path_);

    void load_train_subjects_names();
    void load_test_subjects_names();
    void train_data_split(int split_ratio, int selection);
    void empty_split();

    int get_number_of_training();
    int get_number_of_development();
    int get_number_of_validation();
    int get_number_of_evaluation();
    int get_number_of_testing();

    std::string get_train_subject_name(int position);
    std::string get_develop_subject_name(int position);
    std::string get_valid_subject_name(int position);
    std::string get_eval_subject_name(int position);
    std::string get_test_subject_name(int position);

    void get_train_paths(const std::string subject_name,
                         std::string &volume_path,
                         std::string &segmentation_path);

    void get_train_volume_path(const std::string subject_name,
                               std::string &volume_path);

    void get_train_segmentation_path(const std::string subject_name,
                                     std::string &segmentation_path);

    void get_train_meta_segmentation_path(const std::string subject_name,
                                          std::string &meta_segment_path);

    void get_test_volume_path(const std::string subject_name,
                              std::string &volume_path);

    void get_test_segmentation_path(const std::string subject_name,
                                    std::string &segmentation_path);
};

#endif /* LITS_DATABASE_H_ */

