#include "lits_database.h"

// Training batches of the database
// The first contains 28 subjects and the second 103
std::string train_db_batches[2] = {"Training Batch 1", "Training Batch 2"};

// Testing batch of the database
// It contains 70 subjects
std::string test_batch = "Test Batch";

// Possible database split ratios (%) into development
// validation and evaluation subsets in percents
unsigned short split_ratios[3][3] = {{60, 20, 20}, {50, 25, 25}, {80, 10, 10}};

// Cross validation parameter
// Number of necessary database splits in order to
// evaluate algorithm on all data in such a way that
// there is no overlap between development, validation
// and evaluation subsets. (e.g. for the first
// split ratio and for the first data splitting
// first 60% of the data is used for development,
// following 20% for validation and 20% for evaluation,
// then we shift data selection for 20%, and repeat
// the same until the algorithm is evaluated on all data)
unsigned short n_dev_valid_eval[3] = {5, 4, 10};

/******************************************************************************
 * LiTS_db constructor: assigning database path
 *
 * Arguments:
 * 		db_path_: path to the directory containing folders
 * 		"Training Batch 1" and "Training Batch 2"
 *
 *****************************************************************************/
LiTS_db::LiTS_db(std::string db_path_)
{
    db_path = db_path_;
    n_train = 0;
    n_dev = 0;
    n_valid = 0;
    n_eval = 0;
    n_test = 0;
}

/******************************************************************************
 * load_train_subjects_names - loading all training subjects' names into
 * the private member training_names (vector of strings) and shuffling
 * their order in order to avoid having the data ordering by the institutions
 * they were acquired in
 *****************************************************************************/
void LiTS_db::load_train_subjects_names()
{
    fs::directory_iterator end_iter;
    std::string subject_name;
    std::vector<std::string>::iterator it;

    for (unsigned int i = 0; i < 2; i++)
    {
        for (fs::directory_iterator dir_iter(
                db_path + "/" + train_db_batches[i]); dir_iter != end_iter;
                ++dir_iter)
        {
            if (!strncmp(basename(dir_iter->path()).c_str(), "volume", 6))
            {
                subject_name = basename(dir_iter->path()).substr(7);
                if (!train_subjects.size())
                    train_subjects.push_back(subject_name);
                else
                {
                    for (it = train_subjects.begin();
                            it < train_subjects.end(); it++)
                    {
                        if (std::atoi((*it).c_str()) > std::atoi(
                                subject_name.c_str()))
                        {
                            train_subjects.insert(it, subject_name);
                            break;
                        }
                    }
                    if (it == train_subjects.end())
                        train_subjects.push_back(subject_name);
                }
            }
        }
    }
    std::srand(0);
    std::random_shuffle(train_subjects.begin(), train_subjects.end());
    n_train = train_subjects.size();
}

/******************************************************************************
 * load_test_subjects_names - loading all testing subjects' names into
 * the private member testing_names (vector of strings)
 *****************************************************************************/
void LiTS_db::load_test_subjects_names()
{

    fs::directory_iterator end_iter;
    std::string subject_name;
    std::vector<std::string>::iterator it;

    for (fs::directory_iterator dir_iter(db_path + "/" + test_batch);
            dir_iter != end_iter; ++dir_iter)
    {
        subject_name = basename(dir_iter->path()).substr(12);
        if (!test_subjects.size())
            test_subjects.push_back(subject_name);
        else
        {
            for (it = test_subjects.begin(); it < test_subjects.end(); it++)
            {
                if (std::atoi((*it).c_str()) > std::atoi(subject_name.c_str()))
                {
                    test_subjects.insert(it, subject_name);
                    break;
                }
            }
            if (it == test_subjects.end())
                test_subjects.push_back(subject_name);
        }
    }
    n_test = test_subjects.size();
}

/******************************************************************************
 * train_data_split: splitting data into development, validation and
 * evaluation parts
 *
 * Arguments:
 * 		sr: selection of one out of three offered split ratios
 * 		s: selection of the ordinal number of the data split
 * 		    for the selected split ratio
 *****************************************************************************/
void LiTS_db::train_data_split(int sr, int s)
{
    empty_split();
    int n = float(n_train) / n_dev_valid_eval[sr];
    n_dev = split_ratios[sr][0] / (100 / n_dev_valid_eval[sr]) * n;
    n_valid = split_ratios[sr][1] / (100 / n_dev_valid_eval[sr]) * n;
    n_eval = n_train - n_dev - n_valid;
    for (unsigned int i = 0; i < n_dev; i++)
        develop_subjects.push_back(train_subjects.at((s * n_eval + i) %
                                                     n_train));
    for (unsigned int i = 0; i < n_valid; i++)
        valid_subjects.push_back(train_subjects.at((s * n_eval + n_dev + i) %
                                                   n_train));
    for (unsigned int i = 0; i < n_eval; i++)
        eval_subjects.push_back(train_subjects.at((s * n_eval + n_dev +
                                                   n_valid + i) % n_train));
}

/******************************************************************************
 * empty_split: reseting the number and emptying lists of
 * development, validation and evaluation subjects
 *****************************************************************************/
void LiTS_db::empty_split()
{
    n_dev = 0;
    n_valid = 0;
    n_eval = 0;
    develop_subjects.clear();
    valid_subjects.clear();
    eval_subjects.clear();
}

/******************************************************************************
 * get_number_of_training: returning the total number of subjects in
 * the training subset
 *****************************************************************************/
int LiTS_db::get_number_of_training()
{
    return n_train;
}

/******************************************************************************
 * get_number_of_development: returning the total number of subjects in
 * the development subset
 *****************************************************************************/
int LiTS_db::get_number_of_development()
{
    return n_dev;
}

/******************************************************************************
 * get_number_of_validation: returning the total number of subjects in
 * the validation subset
 *****************************************************************************/
int LiTS_db::get_number_of_validation()
{
    return n_valid;
}

/******************************************************************************
 * get_number_of_evaluation: returning the total number of subjects in
 * the evaluation subset
 *****************************************************************************/
int LiTS_db::get_number_of_evaluation()
{
    return n_eval;
}

/******************************************************************************
 * get_number_of_testing: returning the total number of subjects in
 * the testing subset
 *****************************************************************************/
int LiTS_db::get_number_of_testing()
{
    return n_test;
}

/******************************************************************************
 * get_train_subject_name: returning the subject's name at the required
 * position in the training subset
 *
 * Arguments:
 * 		position: required subject's position in the training database
 *****************************************************************************/
std::string LiTS_db::get_train_subject_name(int position)
{
    if (position < n_train and position >= 0)
        return train_subjects.at(position);
    else
    {
        std::cout << "Position:" << position << std::endl;
        std::cout << "warning: Invalid element selection" << std::endl;
        return NULL;
    }
}

/******************************************************************************
 * get_develop_subject_name: returning the subject's name at the required
 * position in the development subset
 *
 * Arguments:
 * 		position: required subject's position in the development subset
 *****************************************************************************/
std::string LiTS_db::get_develop_subject_name(int position)
{
    if (position < n_dev and position >= 0)
        return develop_subjects.at(position);
    else
    {
        std::cout << "Position:" << position << std::endl;
        std::cout << "warning: Invalid element selection" << std::endl;
        return NULL;
    }
}

/******************************************************************************
 * get_valid_subject_name: returning the subject's name at the required
 * position in the validation subset
 *
 * Arguments:
 * 		position: required subject's position in the validation subset
 *****************************************************************************/
std::string LiTS_db::get_valid_subject_name(int position)
{
    if (position < n_valid and position >= 0)
        return valid_subjects.at(position);
    else
    {
        std::cout << "Position:" << position << std::endl;
        std::cout << "warning: Invalid element selection" << std::endl;
        return NULL;
    }
}

/******************************************************************************
 * get_eval_subject_name: returning the subject's name at the required
 * position in the evaluation subset
 *
 * Arguments:
 * 		position: required subject's position in the evaluation subset
 *****************************************************************************/
std::string LiTS_db::get_eval_subject_name(int position)
{
    if (position < n_eval and position >= 0)
        return eval_subjects.at(position);
    else
    {
        std::cout << "Position:" << position << std::endl;
        std::cout << "warning: Invalid element selection" << std::endl;
        return NULL;
    }
}

/******************************************************************************
 * get_test_subject_name: returning the subject's name at the required
 * position in the testing database
 *
 * Arguments:
 * 		position: required subject's position in the testing database
 *****************************************************************************/
std::string LiTS_db::get_test_subject_name(int position)
{
    if (position < n_test and position >= 0)
        return test_subjects.at(position);
    else
    {
        std::cout << "Position:" << position << std::endl;
        std::cout << "warning: Invalid element selection" << std::endl;
        return NULL;
    }
}

/******************************************************************************
 * get_train_paths: creating the volume and segmentation path for the
 * required subject's name
 *
 * Arguments:
 * 		subj_name: string containing name of the training subject
 * 		volume_path: string where the volume path would be stored
 * 		segment_path: string where the segmentation path would be stored
 *****************************************************************************/
void LiTS_db::get_train_paths(const std::string subj_name,
                              std::string &volume_path,
                              std::string &segment_path)
{
    std::string db_batch;

    if (atoi(subj_name.c_str()) < 28)
        db_batch = "/Training Batch 1";
    else
        db_batch = "/Training Batch 2";

    volume_path = db_path + db_batch + "/volume-" + subj_name + ".nii";

    segment_path = db_path + db_batch + "/segmentation-" + subj_name + ".nii";
}

/******************************************************************************
 * get_train_volume_path: creating the volume path for the
 * required subject's name
 *
 * Arguments:
 *      subj_name: string containing training subject's name
 *      volume_path: string where the volume path would be stored
 ******************************************************************************/
void LiTS_db::get_train_volume_path(const std::string subj_name,
                                    std::string &volume_path)
{
    std::string db_batch;

    if (atoi(subj_name.c_str()) < 28)
        db_batch = "/Training Batch 1";
    else
        db_batch = "/Training Batch 2";

    volume_path = db_path + db_batch + "/volume-" + subj_name + ".nii";
}

/******************************************************************************
 * get_train_segment_path: creating the segmentation path for the
 * required subject's name
 *
 * Arguments:
 *      subject_name: string containing training subject's name
 *      segment_path: string where the segmentation path would be stored
 *****************************************************************************/
void LiTS_db::get_train_segment_path(const std::string subj_name,
                                     std::string &segment_path)
{
    std::string db_batch;

    if (atoi(subj_name.c_str()) < 28)
        db_batch = "/Training Batch 1";
    else
        db_batch = "/Training Batch 2";

    segment_path = db_path + db_batch + "/segmentation-" + subj_name + ".nii";
}

/******************************************************************************
 * get_train_meta_segment_path: creating the segmentation path for
 * the meta segmentation for the required subject's name
 *
 * Arguments:
 *      subj_name: string containing training subject's name
 *      meta_segment_path: string where the meta segmentation path would
 *      be stored
 *****************************************************************************/
void LiTS_db::get_train_meta_segment_path(const std::string subj_name,
                                          std::string &meta_segment_path)
{
    meta_segment_path = db_path + "/Training Meta Segmentations" +
                        "/meta-segment-" + subj_name + ".nii";
}

/******************************************************************************
 * get_test_volume_path: creating the volume path for the
 * required subject's name
 *
 * Arguments:
 *      subject_name: string containing testing subject's name
 *      volume_path: string where the volume path would be stored
 *****************************************************************************/
void LiTS_db::get_test_volume_path(const std::string subj_name,
                                   std::string &volume_path)
{
    volume_path = db_path + test_batch + "/test-volume-" + subj_name + ".nii";
}

/******************************************************************************
 * get_test_segmentation_path: creating the segmentation path for the
 * required subject's name
 *
 * Arguments:
 *      subj_name: string containing testing subject's name
 *      segment_path: string where the segmentation path would be stored
 *****************************************************************************/
void LiTS_db::get_test_segment_path(const std::string subj_name,
                                    std::string &segment_path)
{
    std::string db_batch = "/Testing Results";
    segment_path =  db_path + db_batch + "/test-segment-" + subj_name + ".nii";
}

