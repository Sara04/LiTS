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
 * LiTS_liver_side_estimator constructor: input data size initialization,
 * slice selection parameters initialization, nn_clf initialization
 * ???? nn_clf initialization should be updated ????
 *
 * Arguments:
 *      w_rs_: input image width
 *      h_rs_: input image height
 *      ext_d_: slice selection extension below lungs bottom
 *      ext_u_: slice selection extension above lungs bottom
 *****************************************************************************/
LiTS_liver_side_estimator::LiTS_liver_side_estimator(unsigned w_rs_,
                                                     unsigned h_rs_,
                                                     float ext_d_,
                                                     float ext_u_)
{
    w_rs = w_rs_;
    h_rs = h_rs_;
    ext_d = ext_d_;
    ext_u = ext_u_;
    N_slices = 0;
    nn_clf_on_gpu = false;

    std::vector<std::string> layers;
    unsigned **W_sizes;
    unsigned **b_sizes;

    layers.push_back("fcn");
    layers.push_back("fcn");
    layers.push_back("fcn");
    layers.push_back("fcn");

    W_sizes = new unsigned *[4];
    b_sizes = new unsigned *[4];

    // 1. Layer
    W_sizes[0] = new unsigned[4];
    b_sizes[0] = new unsigned[1];
    W_sizes[0][0] = w_rs * h_rs;
    W_sizes[0][1] = w_rs * h_rs / 8;
    W_sizes[0][2] = 1;
    W_sizes[0][3] = 1;
    b_sizes[0][0] = w_rs * h_rs / 8;

    // 2. Layer
    W_sizes[1] = new unsigned[4];
    b_sizes[1] = new unsigned[1];
    W_sizes[1][0] = W_sizes[0][1];
    W_sizes[1][1] = W_sizes[0][1] / 8;
    W_sizes[1][2] = 1;
    W_sizes[1][3] = 1;
    b_sizes[1][0] = W_sizes[0][1] / 8;

    // 3. Layer
    W_sizes[2] = new unsigned[4];
    b_sizes[2] = new unsigned[1];
    W_sizes[2][0] = W_sizes[1][1];
    W_sizes[2][1] = W_sizes[1][1] / 8;
    W_sizes[2][2] = 1;
    W_sizes[2][3] = 1;
    b_sizes[2][0] = W_sizes[1][1] / 8;


    // 4. Layer
    W_sizes[3] = new unsigned[4];
    b_sizes[3] = new unsigned[1];
    W_sizes[3][0] = W_sizes[2][1];
    W_sizes[3][1] = 1;
    W_sizes[3][2] = 1;
    W_sizes[3][3] = 1;
    b_sizes[3][0] = 1;

    NN nn_init = NN(layers, W_sizes, b_sizes);
    nn_clf = nn_init;
}

/******************************************************************************
 * ~LiTS_liver_side_estimator destructor
 *****************************************************************************/
LiTS_liver_side_estimator::~LiTS_liver_side_estimator()
{
}

/******************************************************************************
 * create_training_data: create a training batch
 *
 * Arguments:
 *      ts: vector of LiTS_scan objects
 *****************************************************************************/
void LiTS_liver_side_estimator::create_training_data(std::vector<LiTS_scan> ts,
                                                     unsigned N_augment)
{
    // _____________________________________________________________________ //
    // 1. Allocating memory for the training data
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

        S[3 * i] = ts.at(i).get_width();
        S[3 * i + 1] = ts.at(i).get_height();
        S[3 * i + 2] = ts.at(i).get_depth();

        vox_S[3 * i] = ts.at(i).get_voxel_width();
        vox_S[3 * i + 1] = ts.at(i).get_voxel_height();
        vox_S[3 * i + 2] = ts.at(i).get_voxel_depth();

        Ls[i + 1] = Ls[i] + S[3 * i] * S[3 * i + 1] * S[3 * i + 2];

        Vs[i] = ts.at(i).get_volume()->GetBufferPointer();
        masks_gt[i] = ts.at(i).get_segmentation()->GetBufferPointer();
        masks_m[i] = ts.at(i).get_meta_segmentation()-> GetBufferPointer();
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
    // _____________________________________________________________________ //
    // 4. Training slices extraction
    // Training slices extraction based on the estimated lungs bottom bound
    // and given upper and lower extensions (it is assumed that slice distance
    // is correct, what is not the case for all scans)
    // _____________________________________________________________________ //
    training_data = new float[2 * N_sl * N_augment * w_rs * h_rs];
    extract_slices(Vs, training_data, B, S, N_s, N_augment, N_sl, N_pix,
                   w_rs, h_rs, ts_T, ts_B);
    // _____________________________________________________________________ //
    // 5. Extract labels
    // Liver side ground truth label is extracted from the liver ground truth
    // segmentation data
    // _____________________________________________________________________ //
    bool *gt = new bool[N_s];
    extract_liver_side_ground_truth(masks_gt, S, Ls, N_s, B, gt);
    training_gt = new float[2 * N_sl * N_augment];
    unsigned idx = 0;
    for(unsigned int s = 0; s < N_s; s++)
        for(unsigned int a = 0; a < N_augment; a++)
            for(unsigned i = 0; i < (ts_T[s] + 1 - ts_B[s]); i++)
            {
                training_gt[idx] = gt[s];
                training_gt[N_sl * N_augment + idx] = 1 - gt[s];
                idx += 1;
            }
    N_slices = 2 * N_sl * N_augment;

    // 6. Memory release
    delete [] ts_T;
    delete [] ts_B;
    delete [] B;
    delete [] masks_gt;
    delete [] masks_m;
    delete [] S;
    delete [] vox_S;
    delete [] Ls;
}

void LiTS_liver_side_estimator::create_testing_data(
        std::vector<LiTS_scan> test_scans_batch)
{
}

void LiTS_liver_side_estimator::
    train_liver_side_estimator(LiTS_db &db, LiTS_processor &p,
                               unsigned N_iters,
                               unsigned N_subj_batch,
                               unsigned N_augment)
{
    if (!nn_clf_on_gpu)
    {
        nn_clf.transfer_trainable_parameters();
        nn_clf_on_gpu = true;
    }

    boost::progress_display show_progress(N_iters);
    for(unsigned int it = 0; it < N_iters; it++)
    {
        std::cout<<"iteration:"<<it<<std::endl;
        std::vector<LiTS_scan> train_scans_batch;
        for(unsigned int i = 0; i < N_subj_batch; i++)
        {
            unsigned int d_idx = rand() % db.get_number_of_development();
            std::string volume_path;
            std::string segment_path;
            std::string meta_segment_path;
            std::string scan_name;

            scan_name = db.get_develop_subject_name(d_idx);
            db.get_train_paths(scan_name, volume_path, segment_path);
            db.get_train_meta_segmentation_path(scan_name,
                                                meta_segment_path);

            LiTS_scan ls(volume_path, segment_path, meta_segment_path);
            ls.load_volume();
            ls.load_segmentation();
            ls.load_meta_segmentation();
            ls.load_info();

            p.preprocess_volume(&ls);
            p.reorient_segmentation((ls.get_segmentation())
                                    ->GetBufferPointer(),
                                    ls.get_width(),
                                    ls.get_height(),
                                    ls.get_depth(),
                                    ls.get_axes_order(),
                                    ls.get_axes_orientation(),
                                    p.get_axes_order(),
                                    p.get_axes_orientation());
            p.reorient_segmentation((ls.get_meta_segmentation())
                                    ->GetBufferPointer(),
                                    ls.get_width(),
                                    ls.get_height(),
                                    ls.get_depth(),
                                    ls.get_axes_order(),
                                    ls.get_axes_orientation(),
                                    p.get_axes_order(),
                                    p.get_axes_orientation());
            train_scans_batch.push_back(ls);
        }
        create_training_data(train_scans_batch, N_augment);
        unsigned train_S[4] = {w_rs, h_rs, 1, N_slices};
        float training_error = 0.0;
        nn_clf.propagate_forward_train(training_data, train_S);
        nn_clf.propagate_backwards_train(training_gt, train_S);
        //exit(EXIT_FAILURE);

        /*
        float *img_tmp = new float[w_rs * h_rs];
        for(unsigned int i = 0; i < N_slices; i++)
        {
            for(unsigned int j = 0; j < (w_rs * h_rs); j++)
                img_tmp[j] = training_data[i * w_rs * h_rs + j] + 0.5;

            cv::Mat img_cv = cv::Mat(h_rs, w_rs, CV_32F, img_tmp);
            cv::imshow("img cv", img_cv);
            cv::waitKey(0);
        }
        */
        ++show_progress;
    }
}




