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
 * Class LiTS_db for database management.
 * It has means of creating list of all available subjects
 * in the database; splitting data into training, validation
 * and testing parts; getting the number of training, validation
 * and testing subjects; getting the subjects of the training,
 * validation and testing sub-databases; getting the volume and
 * segmentation paths of a scan.
 *
 * Arguments:
 * 		db_path: path to the directory containing folders
 * 		         "Training Batch 1" and "Training Batch 2"
 * 		subject_names: vector for storing all subject/scan names
 * 		training_subjects: vector for storing all training subject/scan
 * 		                   names
 * 		validation_subjects: vector for storing all validation subject/scan
 * 		                     names
 * 		testing_subjects: vector for storing all testing subject/scan names
 * 		n_subjects: the total number of subjects/scans
 * 		n_train: the total number of training subjects/scans
 * 		n_valid: the total number of validation subjects/scans
 * 		n_test: the total number of testing subjects/scans
 *
 * Methods:
 * 		LiTS_db: constructor
 * 		~LiTS_db: destructor
 * 		load_subject_names: loading all subjects'/scans' names
 * 		data_split: splitting data into training, validation and testing parts
 * 		empty_split: reseting data split
 * 		get_number_of_subjects: get the total number of loaded subjects/scans
 * 		get_number_of_training: get the total number of training subjects/
 * 		                        scans
 * 		get_number_of_validation: get the total number of validation subjects/
 * 		                          scans
 * 		get_number_of_testing: get the total number of testing subjects/scans
 * 		get_train_scan_name: get the subject/scan name from the training
 * 		                     subset
 * 		get_valid_scan_name: get the subject/scan name from the validation
 * 		                     subset
 * 		get_test_scan_name: get the subject/scan name from the testing subset
 * 		get_scan_paths: create volume and segmentation paths for a given scan
 *
 */
class LiTS_db
{
private:
	std::string db_path;
	std::vector<std::string> subject_names;
	std::vector<std::string> training_subjects;
	std::vector<std::string> validation_subjects;
	std::vector<std::string> testing_subjects;

	int n_subjects;
	int n_train;
	int n_valid;
	int n_test;

public:


	LiTS_db(std::string db_path_);
	~LiTS_db();

	void load_subject_names();
	void data_split(int split_ratio, int selection);
	void empty_split();

	int get_number_of_subjects();
	int get_number_of_training();
	int get_number_of_validation();
	int get_number_of_testing();

	std::string get_train_scan_name(int position);
	std::string get_valid_scan_name(int position);
	std::string get_test_scan_name(int position);

	void get_scan_paths(const std::string scan_name,
			            std::string volume_path,
			            std::string segmentation_path);
};

#endif /* LITS_DATABASE_H_ */

