#include "lits_database.h"

// Training batches of the database
// First contains 28 subjects and second 103
std::string train_db_batches[2] = {"Training Batch 1", "Training Batch 2"};

// Testing batch of the database
std::string test_batch = "Testing Batch";

// Possible database split ratios into development
// validation and evaluation subsets in percents
unsigned short split_ratios[3][3] = {{60, 20, 20}, {50, 25, 25}, {80, 10, 10}};

// Number of necessary database splits in order to
// evaluate algorithm on all data in such a way that
// there is no overlap between development, validation
// and evaluation subsets. (e.g. for the first
// split ratio and for the first data splitting
// first 60% of the data is used for development,
// following 20% for validation and 20% for evaluation,
// then we shift data selection for 20%, and repeat
// the same until the algorithm is evaluated on all data)
unsigned short n_develop_valid_eval[3] = {5, 4, 10};

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
    n_train = 0;
    n_develop = 0;
    n_valid = 0;
    n_eval = 0;
    n_test = 0;
}

/*
 * load_train_subject_names - loading all training subject names into
 * the private member training_names (vector of strings) and shuffling
 * their order in order to avoid having data ordering by the institutions
 * they were acquired in
 */
void LiTS_db::load_train_subject_names()
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
                if (!training_subjects.size())
                    training_subjects.push_back(subject_name);
                else
                {
                    for (it = training_subjects.begin();
                            it < training_subjects.end(); it++)
                    {
                        if (std::atoi((*it).c_str()) > std::atoi(
                                subject_name.c_str()))
                        {
                            training_subjects.insert(it, subject_name);
                            break;
                        }
                    }
                    if (it == training_subjects.end())
                        training_subjects.push_back(subject_name);
                }
            }
        }
    }
    std::srand(0);
    std::random_shuffle(training_subjects.begin(), training_subjects.end());
    n_train = training_subjects.size();
}

/*
 * load_test_subject_names - loading all testing subject names into
 * the private member testing_names (vector of strings)
 */
void LiTS_db::load_test_subject_names()
{

    fs::directory_iterator end_iter;
    std::string subject_name;
    std::vector<std::string>::iterator it;

    for (fs::directory_iterator dir_iter(db_path + "/" + test_batch);
            dir_iter != end_iter; ++dir_iter)
    {
        subject_name = basename(dir_iter->path()).substr(12);
        if (!testing_subjects.size())
            testing_subjects.push_back(subject_name);
        else
        {
            for (it = testing_subjects.begin(); it < testing_subjects.end();
                    it++)
            {
                if (std::atoi((*it).c_str()) > std::atoi(subject_name.c_str()))
                {
                    testing_subjects.insert(it, subject_name);
                    break;
                }
            }
            if (it == testing_subjects.end())
                testing_subjects.push_back(subject_name);
        }
    }
    n_test = testing_subjects.size();
}
/*
 * train_data_split: splitting data into development, validation and
 * evaluation parts
 *
 * Arguments:
 * 		split_ratio: selection of one of the the three offered split ratios
 * 		selection: selection of the training split ordinal number for the
 * 		given split ratio
 */
void LiTS_db::train_data_split(int split_ratio, int selection)
{
    empty_split();
    int n = float(n_train) / n_develop_valid_eval[split_ratio];
    n_develop = split_ratios[split_ratio][0]
            / (100 / n_develop_valid_eval[split_ratio])
                * n;
    n_valid = split_ratios[split_ratio][1]
            / (100 / n_develop_valid_eval[split_ratio])
              * n;
    n_eval = n_train - n_develop - n_valid;
    for (unsigned int i = 0; i < n_develop; i++)
        development_subjects.push_back(
                training_subjects.at((selection * n_eval + i) % n_train));
    for (unsigned int i = 0; i < n_valid; i++)
        validation_subjects.push_back(
                training_subjects.at(
                        (selection * n_eval + n_develop + i) % n_train));
    for (unsigned int i = 0; i < n_eval; i++)
        evaluation_subjects.push_back(
                training_subjects.at(
                        (selection * n_eval + n_develop + n_valid + i) %
                         n_train));
}

/*
 * empty_split: reseting the number and emptying lists of
 * development, validation and evaluation subjects
 */
void LiTS_db::empty_split()
{
    n_develop = 0;
    n_valid = 0;
    n_eval = 0;
    development_subjects.clear();
    validation_subjects.clear();
    evaluation_subjects.clear();
}

/*
 * get_number_of_training: returning the total number of subjects in
 * the training database
 */
int LiTS_db::get_number_of_training()
{
    return n_train;
}

/*
 * get_number_of_development: returning the total number of subjects in
 * the development subset
 */
int LiTS_db::get_number_of_development()
{
    return n_develop;
}

/*
 * get_number_of_validation: returning the total number of subjects in
 * the validation subset
 */
int LiTS_db::get_number_of_validation()
{
    return n_valid;
}

/*
 * get_number_of_evaluation: returning the total number of subjects in
 * the evaluation subset
 */
int LiTS_db::get_number_of_evaluation()
{
    return n_eval;
}

/*
 * get_number_of_testing: returning the total number of subjects in
 * the testing database
 */
int LiTS_db::get_number_of_testing()
{
    return n_test;
}

/*
 * get_train_subject_name: returning the subject's name at the required
 * position in the training database
 *
 * Arguments:
 * 		position: required subject's position in the training database
 */
std::string LiTS_db::get_train_subject_name(int position)
{
    if (position < n_train and position >= 0)
        return training_subjects.at(position);
    else
    {
        std::cout << "Position:" << position << std::endl;
        std::cout << "warning: Invalid element selection" << std::endl;
        return NULL;
    }
}

/*
 * get_develop_subject_name: returning the subject's name at the required
 * position in the development subset
 *
 * Arguments:
 * 		position: required subject's position in the development subset
 */
std::string LiTS_db::get_develop_subject_name(int position)
{
    if (position < n_develop and position >= 0)
        return development_subjects.at(position);
    else
    {
        std::cout << "Position:" << position << std::endl;
        std::cout << "warning: Invalid element selection" << std::endl;
        return NULL;
    }
}

/*
 * get_valid_subject_name: returning the subject's name at the required
 * position in the validation subset
 *
 * Arguments:
 * 		position: required subject's position in the validation subset
 */
std::string LiTS_db::get_valid_subject_name(int position)
{
    if (position < n_valid and position >= 0)
        return validation_subjects.at(position);
    else
    {
        std::cout << "Position:" << position << std::endl;
        std::cout << "warning: Invalid element selection" << std::endl;
        return NULL;
    }
}

/*
 * get_eval_subject_name: returning the subject's name at the required
 * position in the evaluation subset
 *
 * Arguments:
 * 		position: required subject's position in the evaluation subset
 */
std::string LiTS_db::get_eval_subject_name(int position)
{
    if (position < n_eval and position >= 0)
        return evaluation_subjects.at(position);
    else
    {
        std::cout << "Position:" << position << std::endl;
        std::cout << "warning: Invalid element selection" << std::endl;
        return NULL;
    }
}

/*
 * get_test_subject_name: returning the subject's name at the required
 * position in the testing database
 *
 * Arguments:
 * 		position: required subject's position in the testing database
 */
std::string LiTS_db::get_test_subject_name(int position)
{
    if (position < n_test and position >= 0)
        return testing_subjects.at(position);
    else
    {
        std::cout << "Position:" << position << std::endl;
        std::cout << "warning: Invalid element selection" << std::endl;
        return NULL;
    }
}

/*
 * get_train_paths: creating the volume and segmentation path for the
 * required subject's name
 *
 * Arguments:
 * 		subject_name: string containing name of the training subject
 * 		volume_path: string where the volume path would be stored
 * 		segmentation_path: string where the segmentation path would
 * 			be stored
 */
void LiTS_db::get_train_paths(const std::string subject_name,
                              std::string &volume_path,
                              std::string &segmentation_path)
{
    std::string db_batch;

    if (atoi(subject_name.c_str()) < 28)
        db_batch = "/Training Batch 1";
    else
        db_batch = "/Training Batch 2";

    volume_path = db_path + db_batch + "/volume-" + subject_name + ".nii";

    segmentation_path = db_path + db_batch + "/segmentation-" + subject_name
                        + ".nii";
}

/*
 * get_test_path: creating the volume path for the required subject's name
 *
 * Arguments:
 * 		subject_name: string containing name of the testing subject
 * 		volume_path: string where the volume path would be stored
 */
void LiTS_db::get_test_path(const std::string subject_name,
                            std::string &volume_path)
{
    volume_path = db_path + "/Testing Batch" + "/test-volume-" + subject_name
                  + ".nii";
}
