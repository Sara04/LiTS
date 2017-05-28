/*
 * liver_side_estimator.cpp
 *
 *  Created on: Apr 15, 2017
 *      Author: sara
 */

#include "liver_side_estimator.h"
#include "liver_side_estimator.cuh"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/******************************************************************************
 * LiTS_liver_side_estimator: empty constructor
 *****************************************************************************/
LiTS_liver_side_estimator::LiTS_liver_side_estimator()
{
    mean_std_allocated = false;
}
/******************************************************************************
 * LiTS_liver_side_estimator constructor: input data size initialization,
 * slice selection parameters initialization, nn_clf initialization
 * ???? nn_clf initialization should be updated ????
 *
 * Arguments:
 *      model_path: path to the directory where mean, std and model will be/are
 *          saved
 *      w_rs_: input image width
 *      h_rs_: input image height
 *      ext_d_: slice selection extension below lungs bottom
 *      ext_u_: slice selection extension above lungs bottom
 *****************************************************************************/
LiTS_liver_side_estimator::LiTS_liver_side_estimator(std::string model_path_,
                                                     unsigned w_rs_,
                                                     unsigned h_rs_,
                                                     float ext_d_,
                                                     float ext_u_)
{
    model_path = model_path_;
    w_rs = w_rs_;
    h_rs = h_rs_;
    ext_d = ext_d_;
    ext_u = ext_u_;
    N_slices = 0;
    nn_clf_on_gpu = false;

    mean = new float[w_rs * h_rs];
    std = new float[w_rs * h_rs];
    mean_std_allocated = true;

    NN nn_init;
    if(fs::exists(model_path + "done_train"))
    {
        std::cout<<"Loading saved model!!!"<<std::endl;
        nn_init = NN(model_path);
    }
    else
    {
        std::vector<std::string> layers;
        unsigned **W_sizes;
        unsigned **b_sizes;

        for(unsigned i = 0; i < 4; i++)
            layers.push_back("fcn");

        W_sizes = new unsigned *[4];
        b_sizes = new unsigned *[4];
        for(unsigned i = 0;i < 4; i++)
        {
            W_sizes[i] = new unsigned[4];
            b_sizes[i] = new unsigned[1];
            W_sizes[i][2] = 1;
            W_sizes[i][3] = 1;
        }

        // 1. Layer
        W_sizes[0][0] = w_rs * h_rs;
        W_sizes[0][1] = w_rs * h_rs / 8;
        b_sizes[0][0] = w_rs * h_rs / 8;

        // 2. Layer
        W_sizes[1][0] = W_sizes[0][1];
        W_sizes[1][1] = W_sizes[0][1] / 8;
        b_sizes[1][0] = W_sizes[0][1] / 8;

        // 3. Layer
        W_sizes[2][0] = W_sizes[1][1];
        W_sizes[2][1] = W_sizes[1][1] / 8;
        b_sizes[2][0] = W_sizes[1][1] / 8;

        // 4. Layer
        W_sizes[3][0] = W_sizes[2][1];
        W_sizes[3][1] = 1;
        b_sizes[3][0] = 1;
        nn_init = NN(layers, W_sizes, b_sizes);
    }

    nn_clf = nn_init;
}

/******************************************************************************
 * ~LiTS_liver_side_estimator destructor
 *****************************************************************************/
LiTS_liver_side_estimator::~LiTS_liver_side_estimator()
{
    if(mean_std_allocated)
    {
        delete [] mean;
        delete [] std;
    }
}

/******************************************************************************
 * load_and_preprocess_scan: method that loads volume, segmentation and
 * meta segmentation and pre-process them in terms of voxel intensities and
 * axis order and orientation
 *
 * Arguments:
 *      p: processor
 *      ls: scan containing volume, segmentation and meta segmentation
 *****************************************************************************/
void LiTS_liver_side_estimator::load_and_preprocess_scan(LiTS_processor &p,
                                                         LiTS_scan &ls)
{
    ls.load_volume();
    ls.load_segment();
    ls.load_meta_segment();
    ls.load_info();

    p.preprocess_volume(&ls);
    p.reorient_segment((ls.get_segment())->GetBufferPointer(),
                       ls.get_width(), ls.get_height(), ls.get_depth(),
                       ls.get_axes_order(), ls.get_axes_orient(),
                       p.get_axes_order(), p.get_axes_orient());
    p.reorient_segment((ls.get_meta_segment())->GetBufferPointer(),
                       ls.get_width(), ls.get_height(), ls.get_depth(),
                       ls.get_axes_order(), ls.get_axes_orient(),
                       p.get_axes_order(), p.get_axes_orient());
}

/******************************************************************************
 * get_volume_and_voxel_sizes: method that loads volume's and voxel's sizes
 *
 * Arguments:
 *      ls: scan containing volume, segmentation and meta segmentation
 *      S: array where to store volume's size
 *      vox_S: array where to store voxel's size
 *****************************************************************************/
void LiTS_liver_side_estimator::get_volume_and_voxel_sizes(LiTS_scan &ls,
                                                           unsigned *S,
                                                           float *vox_S)
{
    S[0] = ls.get_width();
    S[1] = ls.get_height();
    S[2] = ls.get_depth();

    vox_S[0] = ls.get_voxel_width();
    vox_S[1] = ls.get_voxel_height();
    vox_S[2] = ls.get_voxel_depth();
}

/******************************************************************************
 * compute_mean: input image mean computation
 *
 * Arguments:
 *      db: LiTS database with defined split into dev, valid, eval
 *      p: processor for voxel value normalization, re-ordering and
 *         re-orienting of axes of volume, segmentation and meta
 *         segmentation
 *
 *****************************************************************************/
void LiTS_liver_side_estimator::compute_mean(LiTS_db &db, LiTS_processor &p)
{
    for(unsigned int i = 0; i < (w_rs * h_rs); i++)
        mean[i] = 0.0;
    unsigned count_slices = 0;
    std::vector<LiTS_scan> develop_scans_batch;
    for(unsigned int i = 0; i < db.get_number_of_development(); i++)
    {
        std::string scan_name = db.get_develop_subject_name(i);
        std::string volume_path, segment_path, meta_segment_path;
        db.get_train_paths(scan_name, volume_path, segment_path);
        db.get_train_meta_segment_path(scan_name, meta_segment_path);
        LiTS_scan ls(volume_path, segment_path, meta_segment_path);
        load_and_preprocess_scan(p, ls);
        develop_scans_batch.push_back(ls);
        if(i and (i % 10 == 0 or i == (db.get_number_of_development() - 1)))
        {
            std::cout<<"Iteration: "<<i<<"/"<<db.get_number_of_development();
            std::cout<<"\r"<<std::flush;
            create_input_data(develop_scans_batch, "develop", 1);
            unsigned S[3] = {w_rs, h_rs, N_slices};
            accumulate_for_mean_gpu(develop_data, mean, S);
            count_slices += N_slices;
            delete [] develop_data;
            delete [] develop_gt;
            N_slices = 0;
        }
        develop_scans_batch.clear();
    }
    for(unsigned int i = 0; i < (w_rs * h_rs); i++)
        mean[i] = mean[i] / count_slices;
}

/******************************************************************************
 * compute_std: input image standard deviation computation
 *
 * Arguments:
 *      db: LiTS database with defined split into dev, valid, eval
 *      p: processor for voxel value normalization, re-ordering and
 *         re-orienting of axes of volume, segmentation and meta
 *         segmentation
 *
 *****************************************************************************/
void LiTS_liver_side_estimator::compute_std(LiTS_db &db, LiTS_processor &p)
{
    for(unsigned int i = 0; i < (w_rs * h_rs); i++)
        std[i] = 0.0;
    unsigned count_slices = 0;
    std::vector<LiTS_scan> train_scans_batch;
    for(unsigned int i = 0; i < db.get_number_of_development(); i++)
    {
        std::string scan_name = db.get_develop_subject_name(i);
        std::string volume_path, segment_path, meta_segment_path;
        db.get_train_paths(scan_name, volume_path, segment_path);
        db.get_train_meta_segment_path(scan_name, meta_segment_path);

        LiTS_scan ls(volume_path, segment_path, meta_segment_path);
        load_and_preprocess_scan(p, ls);
        train_scans_batch.push_back(ls);
        if(i and (i % 10 == 0 or i == (db.get_number_of_development() - 1)))
        {
            std::cout<<"Iteration: "<<i<<"/"<<db.get_number_of_development();
            std::cout<<"\r"<<std::flush;
            create_input_data(train_scans_batch, "develop", 1);
            unsigned S[3] = {w_rs, h_rs, N_slices};
            accumulate_for_std_gpu(develop_data, std, mean, S);
            count_slices += N_slices;
            delete [] develop_data;
            delete [] develop_gt;
            N_slices = 0;
        }
        train_scans_batch.clear();
    }

    for(unsigned int i = 0; i < (w_rs * h_rs); i++)
        std[i] = sqrt(std[i] / count_slices);
}

/******************************************************************************
 * save_mean: saving the array containing mean value of the images from the
 * development subset
 *****************************************************************************/
void LiTS_liver_side_estimator::save_mean()
{
    std::ofstream mean_out(model_path + "mean.bin",
                           std::ios::out |std::ios::binary);
    mean_out.write((char *)mean, sizeof(float) * w_rs * h_rs);
}

/******************************************************************************
 * save_std: saving the array containing standard deviation of the images from
 * the development subset
 *****************************************************************************/
void LiTS_liver_side_estimator::save_std()
{
    std::ofstream std_out(model_path + "std.bin",
                          std::ios::out |std::ios::binary);
    std_out.write((char *)std, sizeof(float) * w_rs * h_rs);
}

/******************************************************************************
 * save_model: saving the trained neural network model
 *****************************************************************************/
void LiTS_liver_side_estimator::save_model()
{

    nn_clf.transfer_trainable_parameters_to_cpu();
    nn_clf.save_model(model_path);
}

/******************************************************************************
 * load_mean: loading the array containing mean value of the images from the
 * development subset
 *****************************************************************************/
void LiTS_liver_side_estimator::load_mean()
{
    std::ifstream mean_out(model_path + "mean.bin",
                           std::ios::in |std::ios::binary);
    mean_out.read((char *)mean, sizeof(float) * w_rs * h_rs);
}

/******************************************************************************
 * load_std: loading the array containing standard deviation of the images from
 * the development subset
 *****************************************************************************/
void LiTS_liver_side_estimator::load_std()
{
    std::ifstream std_out(model_path + "std.bin",
                           std::ios::in |std::ios::binary);
    std_out.read((char *)std, sizeof(float) * w_rs * h_rs);
}

/******************************************************************************
 * create_input_data: create desired batch of input images
 *  for training, validation or testing
 *
 * Arguments:
 *      ts: vector of LiTS_scan objects
 *      mode: train/valid/test
 *      N_augment: data augmentation used only in training mode
 *****************************************************************************/
void LiTS_liver_side_estimator::create_input_data(std::vector<LiTS_scan> ts,
                                                  std::string mode,
                                                  unsigned N_augment)
{
    // _____________________________________________________________________ //
    // 1. Allocating memory for the input data
    // _____________________________________________________________________ //
    // N_s - number of training scans
    // Vs - pointer to the array of pointers to the volume data
    // masks_gt - pointer to the array of pointers to the ground truth
    //      segmentations
    // masks_m - pointer to the array of pointers to the meta segmentations
    // S - pointer to the array of volumes'/segmentations' sizes
    // vox_S - pointer to the array of volumes'/segmentations' voxel sizes
    // Ls - pointer to the array for the accumulation of the number of voxels
    //      as training scans are being traversed
    // _____________________________________________________________________ //
    unsigned int N_s = ts.size();
    float ** Vs = new float*[N_s];
    unsigned char ** masks_gt = new unsigned char*[N_s];
    unsigned char ** masks_m = new unsigned char*[N_s];
    unsigned int *S = new unsigned int[3 * N_s];
    float *vox_S = new float[3 * N_s];
    unsigned int *Ls = new unsigned int[N_s + 1];

    for(unsigned int i = 0; i < ts.size(); i++)
    {
        if(!i)
            Ls[i] = 0;
        get_volume_and_voxel_sizes(ts.at(i), &S[3 * i], &vox_S[3 * i]);
        Ls[i + 1] = Ls[i] + S[3 * i] * S[3 * i + 1] * S[3 * i + 2];
        Vs[i] = ts.at(i).get_volume()->GetBufferPointer();
        masks_gt[i] = ts.at(i).get_segment()->GetBufferPointer();
        masks_m[i] = ts.at(i).get_meta_segment()-> GetBufferPointer();
    }
    // _____________________________________________________________________ //
    // 2. Detect lung bounds
    // Lungs bounds estimation based on meta segmentation
    // Accumulation of the lungs' mask is performed along each of the three
    // axes and the bounds are determined from those accumulations
    // _____________________________________________________________________ //
    unsigned int *B = new unsigned int[N_s * 6];
    extract_lung_bounds(masks_m, S, Ls, N_s, B);
    // _____________________________________________________________________ //
    // 3. Count the number of training slices and the number of pixels that
    // will be extracted from the training volumes
    // _____________________________________________________________________ //
    unsigned int N_sl = 0;
    unsigned int N_pix = 0;
    unsigned int *ts_T = new unsigned int[N_s];
    unsigned int *ts_B = new unsigned int[N_s];
    for(unsigned int s = 0; s < N_s; s++)
    {
        if((int(B[6 * s + 4]) - int(ext_d / vox_S[3 * s + 2])) >=0)
            ts_B[s] = int(B[6 * s + 4]) - int(ext_d / vox_S[3 * s + 2]);
        else
            ts_B[s] = 0;
        if((B[6 * s + 4] + int(ext_u / vox_S[3 * s + 2])) < S[3 * s + 2])
            ts_T[s] = B[6 * s + 4] + int(ext_u / vox_S[3 * s + 2]);
        else
            ts_T[s] = S[3 * s + 2] - 1;
        N_sl += (ts_T[s] + 1 - ts_B[s]);
        N_pix += ((ts_T[s] + 1 - ts_B[s]) * S[3 * s] * S[3 * s + 1]);
    }

    bool *gt = new bool[N_s];
    if(!strcmp(mode.c_str(), "develop") or !strcmp(mode.c_str(), "valid") or
       !strcmp(mode.c_str(), "eval"))
        extract_liver_side_ground_truth(masks_gt, S, Ls, N_s, B, gt);

    // _____________________________________________________________________ //
    // 4. Training slices extraction
    // Training slices extraction based on the estimated lungs bottom bound
    // and given upper and lower extensions (it is assumed that slice distance
    // is correct, what is not the case for all scans)
    // _____________________________________________________________________ //
    if(!strcmp(mode.c_str(), "develop"))
    {
        develop_data = new float[2 * N_sl * N_augment * w_rs * h_rs];
        extract_slices(Vs, develop_data, B, S, N_s, N_augment, N_sl, N_pix,
                       w_rs, h_rs, ts_T, ts_B, 10);
        develop_gt = new float[2 * N_sl * N_augment];
        unsigned idx = 0;
        for(unsigned int s = 0; s < N_s; s++)
            for(unsigned int a = 0; a < N_augment; a++)
                for(unsigned i = 0; i < (ts_T[s] + 1 - ts_B[s]); i++)
                {
                    develop_gt[idx] = gt[s];
                    develop_gt[N_sl * N_augment + idx] = 1 - gt[s];
                    idx += 1;
                }
    }
    else if(!strcmp(mode.c_str(), "valid"))
    {
        validate_data = new float[2 * N_sl * N_augment * w_rs * h_rs];
        extract_slices(Vs, validate_data, B, S, N_s, N_augment, N_sl, N_pix,
                       w_rs, h_rs, ts_T, ts_B, 10);
        validate_gt = new float[2 * N_sl * N_augment];
        unsigned idx = 0;
        for(unsigned int s = 0; s < N_s; s++)
            for(unsigned int a = 0; a < N_augment; a++)
                for(unsigned i = 0; i < (ts_T[s] + 1 - ts_B[s]); i++)
                {
                    validate_gt[idx] = gt[s];
                    validate_gt[N_sl * N_augment + idx] = 1 - gt[s];
                    idx += 1;
                }
    }
    else if(!strcmp(mode.c_str(), "eval"))
    {
        eval_data = new float[2 * N_sl * N_augment * w_rs * h_rs];
        extract_slices(Vs, eval_data, B, S, N_s, N_augment, N_sl, N_pix,
                       w_rs, h_rs, ts_T, ts_B, 10);
        eval_gt = new float[2 * N_sl * N_augment];
        unsigned idx = 0;
        for(unsigned int s = 0; s < N_s; s++)
            for(unsigned int a = 0; a < N_augment; a++)
                for(unsigned i = 0; i < (ts_T[s] + 1 - ts_B[s]); i++)
                {
                    eval_gt[idx] = gt[s];
                    eval_gt[N_sl * N_augment + idx] = 1 - gt[s];
                    idx += 1;
                }
    }
    else if(!strcmp(mode.c_str(), "test"))
    {
        test_data = new float[2 * N_sl * w_rs * h_rs];
        extract_slices(Vs, test_data, B, S, N_s, N_augment, N_sl, N_pix,
                       w_rs, h_rs, ts_T, ts_B, 10);
    }
    N_slices = 2 * N_sl * N_augment;
    // 6. Memory release
    delete [] ts_T;
    delete [] ts_B;
    delete [] B;
    delete [] Vs;
    delete [] masks_gt;
    delete [] masks_m;
    delete [] S;
    delete [] vox_S;
    delete [] Ls;
    delete [] gt;
}

/******************************************************************************
 * develop_liver_side_estimator: train neural network model for liver side
 * estimation
 *
 * Arguments:
 *      db: LiTS_db database
 *      p: initialized pre-processor
 *      N_iters: number of training iterations
 *      N_subj_batch: number of subjects per training batch
 *      N_augment: factor of data augmentation
 *      learning_rate: learning rate used in backpropagation algorithm
 *      normalize: flag indication whether to perform mean-std normalization
 *      of the data
 *****************************************************************************/
float LiTS_liver_side_estimator::
    develop_liver_side_estimator(LiTS_db &db, LiTS_processor &p,
                                 unsigned N_iters,
                                 unsigned N_subj_batch,
                                 unsigned N_augment,
                                 float learning_rate,
                                 bool normalize)
{
    if (!nn_clf_on_gpu)
    {
        nn_clf.transfer_trainable_parameters_to_gpu();
        nn_clf_on_gpu = true;

        if(!fs::exists(model_path + "mean.bin"))
        {
            std::cout<<"Mean value computation.."<<std::endl;
            compute_mean(db, p);
            save_mean();
        }
        else
            load_mean();

        if(!fs::exists(model_path + "std.bin"))
        {
            std::cout<<"Standard deviation value computation.."<<std::endl;
            compute_std(db, p);
            save_std();
        }
        else
            load_std();
    }

    std::vector<LiTS_scan> develop_scans_batch;
    for(unsigned int i = 0; i < N_subj_batch; i++)
    {
        unsigned int d_idx = rand() % db.get_number_of_development();
        std::string scan_name = db.get_develop_subject_name(d_idx);
        std::string volume_path, segment_path, meta_segment_path;
        db.get_train_paths(scan_name, volume_path, segment_path);
        db.get_train_meta_segment_path(scan_name, meta_segment_path);
        LiTS_scan ls(volume_path, segment_path, meta_segment_path);
        load_and_preprocess_scan(p, ls);
        develop_scans_batch.push_back(ls);
    }
    create_input_data(develop_scans_batch, "develop", N_augment);
    unsigned dev_S[4] = {w_rs, h_rs, 1, N_slices};
    if (normalize)
        normalize_data(develop_data, mean, std, dev_S);
    nn_clf.propagate_forward_train(develop_data, dev_S);

    float train_p = nn_clf.propagate_backwards_train(develop_gt, dev_S,
                                                     learning_rate);
    delete [] develop_data;
    delete [] develop_gt;

    return train_p;
}

/******************************************************************************
 * valid_liver_side_estimator: validate neural network model for liver side
 * estimation
 *
 * Arguments:
 *      db: LiTS_db database
 *      p: initialized pre-processor
 *      N_subj_batch: number of subjects per validation batch
 *      normalize: flag indication whether to perform mean-std normalization
 *      of the data
 *****************************************************************************/
float LiTS_liver_side_estimator::
    valid_liver_side_estimator(LiTS_db &db, LiTS_processor &p,
                               unsigned N_subj_batch, bool normalize)
{
    if (!nn_clf_on_gpu)
    {
        nn_clf.transfer_trainable_parameters_to_gpu();
        nn_clf_on_gpu = true;
        load_mean();
        load_std();
    }

    std::vector<LiTS_scan> valid_scans_batch;
    for(unsigned int i = 0; i < N_subj_batch; i++)
    {
        unsigned int d_idx = rand() % db.get_number_of_validation();
        std::string scan_name = db.get_valid_subject_name(d_idx);
        std::string volume_path, segment_path, meta_segment_path;
        db.get_train_paths(scan_name, volume_path, segment_path);
        db.get_train_meta_segment_path(scan_name, meta_segment_path);
        LiTS_scan ls(volume_path, segment_path, meta_segment_path);
        load_and_preprocess_scan(p, ls);
        valid_scans_batch.push_back(ls);
    }
    create_input_data(valid_scans_batch, "valid", 1);
    unsigned valid_S[4] = {w_rs, h_rs, 1, N_slices};
    if (normalize)
        normalize_data(validate_data, mean, std, valid_S);
    nn_clf.propagate_forward_train(validate_data, valid_S);

    float valid_p = nn_clf.compute_error(validate_gt, valid_S);

    delete [] validate_data;
    delete [] validate_gt;

    return valid_p;
}

/******************************************************************************
 * eval_liver_side_estimator: evaluate neural network model for liver side
 * estimation
 *
 * Arguments:
 *      db: LiTS_db database
 *      p: initialized pre-processor
 *      scan_name: name of the scan to be evaluated
 *      normalize: flag indication whether to perform mean-std normalization
 *      of the data
 *****************************************************************************/
float LiTS_liver_side_estimator::
    eval_liver_side_estimator(LiTS_db &db,
                              LiTS_processor &p,
                              std::string scan_name,
                              bool normalize)
{
    if (!nn_clf_on_gpu)
    {
        nn_clf.transfer_trainable_parameters_to_gpu();
        nn_clf_on_gpu = true;
        load_mean();
        load_std();
    }

    std::vector<LiTS_scan> eval_scan_batch;

    std::string volume_path, segment_path, meta_segment_path;
    db.get_train_paths(scan_name, volume_path, segment_path);
    db.get_train_meta_segment_path(scan_name, meta_segment_path);
    LiTS_scan ls(volume_path, segment_path, meta_segment_path);
    load_and_preprocess_scan(p, ls);
    eval_scan_batch.push_back(ls);

    create_input_data(eval_scan_batch, "eval", 10);

    unsigned eval_S[4] = {w_rs, h_rs, 1, N_slices};
    if (normalize)
        normalize_data(eval_data, mean, std, eval_S);

    unsigned out_len = nn_clf.get_bias_sizes()[(nn_clf.get_layers()).size() -1][0];
    float *eval_scores = new float[N_slices * out_len];
    nn_clf.propagate_forward_test(eval_data, eval_S, eval_scores);

    float side = 0;
    for(unsigned int i = 0; i < N_slices/2; i++)
    {
            side += eval_scores[i] > 0.5;
            side += eval_scores[N_slices / 2 + i] <= 0.5;
    }

    side /= N_slices;
    std::cout<<"ground truth:"<<eval_gt[0]<<std::endl;
    std::cout<<"estimated:"<<(side > 0.5)<<std::endl;
    std::cout<<std::endl;

    delete [] eval_data;
    delete [] eval_gt;
    delete [] eval_scores;

    return 0;
}

/******************************************************************************
 * estimate_liver_side: estimate liver's side
 *
 * Arguments:
 *      db: LiTS_db database
 *      p: initialized pre-processor
 *      scan_name: name of the scan to be evaluated
 *      normalize: flag indication whether to perform mean-std normalization
 *      of the data
 *****************************************************************************/
float LiTS_liver_side_estimator::estimate_liver_side(LiTS_db &db,
                                                     LiTS_processor &p,
                                                     std::string scan_name,
                                                     bool normalize)
{
    if (!nn_clf_on_gpu)
    {
        nn_clf.transfer_trainable_parameters_to_gpu();
        nn_clf_on_gpu = true;
        load_mean();
        load_std();
    }

    std::vector<LiTS_scan> test_scan_batch;

    std::string volume_path, segment_path;
    db.get_test_volume_path(scan_name, volume_path);
    db.get_test_segment_path(scan_name, segment_path);

    LiTS_scan ls(volume_path, segment_path);
    load_and_preprocess_scan(p, ls);
    test_scan_batch.push_back(ls);

    create_input_data(test_scan_batch, "test", 10);
    unsigned test_S[4] = {w_rs, h_rs, 1, N_slices};
    if (normalize)
        normalize_data(test_data, mean, std, test_S);

    unsigned out_len = nn_clf.get_bias_sizes()[(nn_clf.get_layers()).size() -1][0];
    float *test_scores = new float[N_slices * out_len];
    nn_clf.propagate_forward_test(test_data, test_S, test_scores);

    delete [] test_data;
    delete [] test_scores;
    return 0;

}
