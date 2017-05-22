
/******************************************************************************
 * This is a script that runs following modules:
 * - lung mask segmentation and saving
 * - liver side estimation model training, validation and saving
 * - liver segmentation model training, validation and saving
 * - tumor segmentation model training, validation and saving
 *
 * It should be run in the following way:
 * ./toolchain DATABASE_PATH (directory where Training Batch 1,
 * Training Batch 2 and Testing Batch)
 *****************************************************************************/

#include "toolchain.h"

int main(int argc, char **argv)
{
    /*.......................................................................*/
    /* 1. Input path verification
    /*.......................................................................*/
    if(argc != 2)
    {
        std::cout<<"Invalid number of input arguments."<<std::endl;
        std::cout<<"First argument: directory path where the ";
        std::cout<<"Train Batch 1 and Train Batch 2 are placed."<<std::endl;
        exit(EXIT_FAILURE);
    }

    if(!strcmp(argv[1], "-h") or !strcmp(argv[1], "-help") or
       !strcmp(argv[1], "help") or !strcmp(argv[1], "h"))
    {
        std::cout<<"First argument: directory path where the ";
        std::cout<<"Train Batch 1 and Train Batch 2 are placed."<<std::endl;
        exit(EXIT_SUCCESS);
    }

    fs::path db_path(argv[1]);

    if(fs::is_directory(db_path))
    {
        fs::path db_path_batch((db_path.string() + "/Training Batch 1").c_str());
        if(!fs::is_directory(db_path_batch))
        {
            std::cout<<"Training Batch 1 does not exists."<<std::endl;
            exit(EXIT_FAILURE);
        }
        db_path_batch = db_path.string() + "/Training Batch 2";
        if(!fs::is_directory(db_path_batch))
        {
            std::cout<<"Training Batch 2 does not exists."<<std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cout<<"Invalid database path!"<<std::endl;
        exit(EXIT_FAILURE);
    }

    LiTS_db db(db_path.string());
    db.load_train_subjects_names();
    /*.......................................................................*/
    /* 2. Lung segmentation and saving (if not done)
    /*.......................................................................*/
    fs::path done_path_lungs(db_path.string() +
                             "/Training Meta Segmentations/" + "lungs_done");
    if(fs::exists(done_path_lungs))
        std::cout<<"Lungs masks already segmented!"<<std::endl;
    else
    {
        std::cout<<"\nRunning lung segmentation on training data...\n";
        run_lungs_segmentation(db);
    }
    /*.......................................................................*/
    /* 3. Body left - right estimation development and validation
    /*.......................................................................*/
    fs::path done_path_liver_position(db_path.string() +
                             "/Trained Models/" + "liver_position_done");
    if(fs::exists(done_path_liver_position))
        std::cout<<"Liver position (left-right) classifier done!"<<std::endl;
    else
    {
        std::cout<<"\nTraining and validation of liver side classifier...\n";
        db.train_data_split(0,0);
        liver_side_estimator_train_and_valid(db);
    }

    return 0;
}
