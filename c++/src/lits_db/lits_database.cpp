#include "lits_database.h"

// Training batches of the database
// First contains 28 subjects and second 103
std::string db_batches[2] = {"Training Batch 1",
		                     "Training Batch 2"};

// Possible database split ratios in training
// validation and test sub-databases in percents
unsigned short split_ratios[3][3] = {{60, 20, 20},
		                             {50, 25, 25},
		                             {80, 10, 10}};

// Number of necessary database splits in order to
// test algorithm on all data in such a way that
// there is no overlap between training, validation
// and testing sub-databases. (e.g. for the first
// split ratio and for the first data splitting
// first 60% of the data is used for training,
// following 20% for validation and 20% for test,
// then we shift data selection for 20%, and repeat
// the same until the algorithm is tested on all data)
unsigned short n_train_valid_test[3] = {5, 4, 10};

/*
 * LiTS_db constructor: assigning database path
 *
 * Arguments:
 * 		db_path_: path to the directory containing folders
 * 		"Training Batch 1" and "Training Batch 2"
 *
 */
LiTS_db::LiTS_db(std::string db_path_)
{
	db_path = db_path_;
	n_subjects = 0;
	n_train = 0;
	n_valid = 0;
	n_test = 0;
}

/*
 * ~LiTS_db destructor - deleting/reseting all class attributes
 * and emptying all variables used for subject names storing
 */
LiTS_db::~LiTS_db()
{
	db_path = "";
	n_subjects = 0;
	subject_names.clear();
	empty_split();
}

/*
 * load_subject_names - loading all subject names into the private member
 * subject_names (vector of strings) and shuffling their order in order to
 * avoid having data ordered by the institutions they were acquired in
 */
void LiTS_db::load_subject_names()
{
	fs::directory_iterator end_iter;
	std::string subject_name;
	std::vector<std::string>::iterator it;

	for(unsigned int i = 0; i < 2; i++)
	{
		for(fs::directory_iterator dir_iter(db_path + "/" + db_batches[i]);
			dir_iter != end_iter;
			++dir_iter)
		{
			if (!strncmp(basename(dir_iter->path()).c_str(), "volume", 6))
			{
				subject_name = basename(dir_iter->path());
				subject_name = subject_name.substr(subject_name.find("-") + 1);
				if(!subject_names.size())
					subject_names.push_back(subject_name);
				else
				{
					for(it=subject_names.begin();
						it < subject_names.end();
						it++)
					{
						if(std::atoi((*it).c_str()) >
						   std::atoi(subject_name.c_str()))
						{
							subject_names.insert(it, subject_name);
							break;
						}
					}
					if(it == subject_names.end())
						subject_names.push_back(subject_name);
				}
			}
		}
	}
	std::srand(0);
	std::random_shuffle(subject_names.begin(), subject_names.end());
	n_subjects =  subject_names.size();
}

/*
 * data_split: splitting data into training, validation and testing parts
 *
 * Arguments:
 * 		split_ratio: selection of one of the the three offered split ratios
 * 		selection: selection of the training split ordinal number for the
 * 		given split ratio
 */
void LiTS_db::data_split(int split_ratio, int selection)
{
	empty_split();
	int n = float(subject_names.size()) / n_train_valid_test[split_ratio];
	n_train = split_ratios[split_ratio][0] /
			  (100 / n_train_valid_test[split_ratio]) * n;
	n_valid = split_ratios[split_ratio][1] /
			  (100 / n_train_valid_test[split_ratio]) * n;
	n_test = subject_names.size() - n_train - n_valid;
	for(unsigned int i = 0; i < n_train; i++)
		training_subjects.push_back(subject_names.
				at((selection * n_test + i) % n_subjects));
	for(unsigned int i = 0; i < n_valid; i++)
		validation_subjects.push_back(subject_names.
				at((selection * n_test + n_train + i) % n_subjects));
	for(unsigned int i = 0; i < n_test; i++)
		testing_subjects.push_back(subject_names.
				at((selection * n_test + n_train + n_valid + i) % n_subjects));
}

/*
 * empty_split: reseting the number of training, validation and testing scans
 * and emptying vectors of training, validation and testing scans
 */
void LiTS_db::empty_split()
{
	n_train = 0;
	n_valid = 0;
	n_test = 0;
	training_subjects.clear();
	validation_subjects.clear();
	testing_subjects.clear();
}

/*
 * get_number_of_subjects: returning the total number of subjects/scans in the
 * database
 */
int LiTS_db::get_number_of_subjects()
{
	return n_subjects;
}

/*
 * get_number_of_training: returning the total number of subjects/scans in the
 * training sub-database
 */
int LiTS_db::get_number_of_training()
{
	return n_train;
}

/*
 * get_number_of_validation: returning the total number of subjects/scans in
 * the validation sub-database
 */
int LiTS_db::get_number_of_validation()
{
	return n_valid;
}

/*
 * get_number_of_testing: returning the total number of subjects/scans in the
 * testing sub-database
 */
int LiTS_db::get_number_of_testing()
{
	return n_test;
}

/*
 * get_train_scan_name: returning the subject's/scan's name at the required
 * position in the training sub-database
 *
 * Arguments:
 * 		position: required subject's/scan's position in the training
 * 		sub-database
 */
std::string LiTS_db::get_train_scan_name(int position)
{
	if (position < n_train and position >= 0)
		return training_subjects.at(position);
	else
	{
		std::cout<<"Position:"<<position<<std::endl;
		std::cout<<"warning: Invalid element selection"<<std::endl;
		return NULL;
	}
}

/*
 * get_valid_scan_name: returning the subject's/scan's name at the required
 * position in the validation sub-database
 *
 * Arguments:
 * 		position: required subject's/scan's position in the validation
 * 		sub-database
 */
std::string LiTS_db::get_valid_scan_name(int position)
{
	if (position < n_valid and position >= 0)
		return validation_subjects.at(position);
	else
	{
		std::cout<<"Position:"<<position<<std::endl;
		std::cout<<"warning: Invalid element selection"<<std::endl;
		return NULL;
	}
}

/*
 * get_test_scan_name: returning the subject's/scan's name at the required
 * position in the testing sub-database
 *
 * Arguments:
 * 		position: required subject's/scan's position in the testing
 * 		sub-database
 */
std::string LiTS_db::get_test_scan_name(int position)
{
	if (position < n_test and position >= 0)
		return testing_subjects.at(position);
	else
	{
		std::cout<<"Position:"<<position<<std::endl;
		std::cout<<"warning: Invalid element selection"<<std::endl;
		return NULL;
	}
}

/*
 * get_scan_paths: creating the volume and segmentation path of the
 * required subjec/scan
 *
 * Arguments:
 * 		scan_name: string containing name of the subject/scan
 * 		volume_path: string where the volume path would be stored
 * 		segmentation_path: string where the segmentation path would be stored
 */
void LiTS_db::get_scan_paths(const std::string scan_name,
                             std::string &volume_path,
                             std::string &segmentation_path)
{
	std::string db_batch;

	if (atoi(scan_name.c_str()) < 28)
		db_batch = "/Training Batch 1";
	else
		db_batch = "/Training Batch 2";

	volume_path = db_path + db_batch +
			      "/volume-" + scan_name + ".nii";

	segmentation_path = db_path + db_batch +
			            "/segmentation-" + scan_name + ".nii";
}
