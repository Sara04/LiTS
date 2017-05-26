/*
 * liver_side_estimator.cu
 *
 *  Created on: Apr 16, 2017
 *      Author: sara
 */

#include "liver_side_estimator.cuh"
#include "../segmentation/segmentator_cuda_kernels.cuh"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define MAX_THREADS 1024

void accumulate_for_mean_gpu(float *input_data, float *acc_mean, unsigned *S)
{
    float *acc_mean_d;
    cudaMalloc((void **)&acc_mean_d, S[0] * S[1] * sizeof(float));
    cudaMemcpy(acc_mean_d, acc_mean, S[0] * S[1] * sizeof(float),
               cudaMemcpyHostToDevice);

    float *input_data_d;
    cudaMalloc((void **)&input_data_d, S[0] * S[1] * S[2] * sizeof(float));
    cudaMemcpy(input_data_d, input_data, S[0] * S[1] * S[2] * sizeof(float),
               cudaMemcpyHostToDevice);

    unsigned n_blocks = S[0] * S[1] * S[2] / MAX_THREADS + 1;
    accumulation_gpu<<<n_blocks, MAX_THREADS>>>(input_data_d,
                                                acc_mean_d,
                                                S[0] * S[1] * S[2],
                                                S[0] * S[1]);

    cudaMemcpy(acc_mean, acc_mean_d, S[0] * S[1] * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(acc_mean_d);
    cudaFree(input_data_d);
}

void accumulate_for_std_gpu(float *input_data, float *acc_std, float *mean, unsigned *S)
{
    float *acc_std_d;
    cudaMalloc((void **)&acc_std_d, S[0] * S[1] * sizeof(float));
    cudaMemcpy(acc_std_d, acc_std, S[0] * S[1] * sizeof(float),
               cudaMemcpyHostToDevice);

    float *mean_d;
    cudaMalloc((void **)&mean_d, S[0] * S[1] * sizeof(float));
    cudaMemcpy(mean_d, mean, S[0] * S[1] * sizeof(float),
               cudaMemcpyHostToDevice);

    float *input_data_d;
    cudaMalloc((void **)&input_data_d, S[0] * S[1] * S[2] * sizeof(float));
    cudaMemcpy(input_data_d, input_data, S[0] * S[1] * S[2] * sizeof(float),
               cudaMemcpyHostToDevice);

    unsigned n_blocks = S[0] * S[1] * S[2] / MAX_THREADS + 1;
    squared_accumulation_gpu<<<n_blocks, MAX_THREADS>>>(input_data_d,
                                                        acc_std_d,
                                                        mean_d,
                                                        S[0] * S[1] * S[2],
                                                        S[0] * S[1]);

    cudaMemcpy(acc_std, acc_std_d, S[0] * S[1] * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(mean_d);
    cudaFree(acc_std_d);
    cudaFree(input_data_d);
}

void determine_bounds(unsigned *accs, unsigned *S, unsigned *B,  unsigned N_s)
{
    unsigned int sh = 0;
    for(unsigned int s = 0; s < N_s; s++)
    {
        for(unsigned j = 0; j < 6; j++)
            B[6 * s + j] = 0;
        for(unsigned j = 0; j < 3; j++)
        {
            for(unsigned int i = 0; i < S[3 * s + j]; i++)
            {
                if(!B[6*s + 2 * j] and accs[sh + i])
                    B[6*s + 2 * j] = i;
                if(!B[6*s + + 2 * j + 1] and accs[sh + S[3 * s + j] - 1 - i])
                    B[6*s + 2 * j + 1] = S[3 * s + j] - 1 - i;
                if(B[6*s + 2 * j] and B[6 * s + 2 * j + 1])
                    break;
            }
            sh += S[3 * s + j];
        }
    }
}

void extract_lung_bounds(unsigned char **masks_m, unsigned *S, unsigned *Ls,
                         unsigned N_s, unsigned *B)
{
    // 1. Determine lung bounds
    unsigned char *masks_m_d;
    unsigned int *S_d;
    unsigned int *Ls_d;
    unsigned int *accs_d;
    unsigned int *accs;

    // 1.1 Allocate and transfer meta data to gpu
    cudaMalloc((void **)&masks_m_d, Ls[N_s] * sizeof(unsigned char));
    unsigned int s = 0;
    for(unsigned int i = 0; i < N_s; i++)
    {
        cudaMemcpy(&masks_m_d[Ls[i]], masks_m[i],
                   (Ls[i+1] - Ls[i]) * sizeof(unsigned char),
                   cudaMemcpyHostToDevice);
        s += (S[3 * i] + S[3 * i + 1] + S[3 * i + 2]);
    }

    // 1.2 Allocate and transfer accumulators to gpu
    accs = new unsigned int[s];
    for(unsigned int i = 0; i < s; i++)
        accs[i] = 0;
    cudaMalloc((void **)&accs_d, s * sizeof(unsigned));
    cudaMemcpy(accs_d, accs, s * sizeof(unsigned), cudaMemcpyHostToDevice);

    // 1.3 Allocate and transfer sizes and lengths to gpu
    cudaMalloc((void **)&S_d, N_s * 3 * sizeof(unsigned));
    cudaMemcpy(S_d, S, N_s * 3 * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&Ls_d, (N_s + 1) * sizeof(unsigned));
    cudaMemcpy(Ls_d, Ls, (N_s + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);

    unsigned int grid_x = 512;
    unsigned int grid_y = Ls[N_s] / (grid_x * MAX_THREADS) + 1;
    dim3 grid(grid_x, grid_y);

    get_organ_mask_accs_gpu_multiple<<<grid, MAX_THREADS>>>(masks_m_d,
                                                            Ls_d, S_d,
                                                            3, accs_d, N_s);

    cudaMemcpy(accs, accs_d, s * sizeof(unsigned), cudaMemcpyDeviceToHost);

    determine_bounds(accs, S, B, N_s);

    delete [] accs;
    cudaFree(Ls_d);
    cudaFree(S_d);
    cudaFree(accs_d);
    cudaFree(masks_m_d);
}

void extract_slices(float **Vs, float *sls_rs, unsigned *B,
                    unsigned *S, unsigned N_s, unsigned N_augment,
                    unsigned N_sl, unsigned N_pix,
                    unsigned w_rs, unsigned h_rs,
                    unsigned *ts_T, unsigned *ts_B)
{
    float *sls_d;
    float *sls_rs_d;
    unsigned int *Ls_bx_d;
    unsigned int *Ls_bx = new unsigned int[N_sl + 1];
    unsigned int *sls_bx = new unsigned int[4 * N_sl * N_augment];
    unsigned int *sls_bx_d;

    cudaMalloc((void **)&sls_d, N_pix * sizeof(float));
    cudaMalloc((void **)&sls_rs_d, 2 * N_sl * N_augment * w_rs * h_rs * sizeof(float));
    cudaMalloc((void **)&sls_bx_d, 4 * N_sl * N_augment * sizeof(unsigned int));
    cudaMalloc((void **)&Ls_bx_d, (N_sl + 1) * sizeof(unsigned int));

    unsigned int inc = 0;
    Ls_bx[0] = 0;
    for(unsigned int s = 0; s < N_s; s ++)
        for(unsigned int i = ts_B[s]; i <= ts_T[s]; i++)
        {
            cudaMemcpy(&sls_d[Ls_bx[inc]], &Vs[s][i * S[3 * s] * S[3 * s + 1]],
                       S[3 * s] * S[3 * s + 1] * sizeof(float),
                       cudaMemcpyHostToDevice);

            Ls_bx[inc + 1] = Ls_bx[inc] + S[3 * s] * S[3 * s + 1];
            for(unsigned int a = 0; a < N_augment; a++)
            {
                for(unsigned int j = 0; j < 4; j++)
                    sls_bx[4 * inc * N_augment + a * 4 + j] = B[6 * s + j] + std::rand() % 10;
            }
            inc += 1;
        }


    cudaMemcpy(sls_bx_d, sls_bx, 4 * N_sl * N_augment * sizeof(unsigned),
               cudaMemcpyHostToDevice);
    cudaMemcpy(Ls_bx_d, Ls_bx, (N_sl + 1) * sizeof(unsigned),
               cudaMemcpyHostToDevice);

    unsigned int no_blocks = (N_sl * N_augment * w_rs * h_rs) / MAX_THREADS + 1;

    resize_slice_and_crop<<<no_blocks, MAX_THREADS>>>(sls_d, sls_rs_d, sls_bx_d,
                                                 Ls_bx_d, N_sl, N_augment,
                                                 w_rs, h_rs);

    /*
    float *imgs = new float[N_sl * N_augment * w_rs * h_rs];
    cudaMemcpy(imgs, sls_rs_d, N_sl * N_augment * w_rs * h_rs * sizeof(float),
               cudaMemcpyDeviceToHost);

    for(unsigned int i = 0; i < N_sl * N_augment; i++)
    {
        float *img_tmp = new float[w_rs * h_rs];
        img_tmp = &imgs[i * w_rs * h_rs];
        for(unsigned j = 0; j < w_rs * h_rs; j++)
            img_tmp[j] = img_tmp[j] + 0.5;

        cv::Mat segment = cv::Mat(h_rs, w_rs, CV_32F, img_tmp);

        cv::namedWindow( "Display window");
        cv::imshow( "Display window", segment);
        cv::waitKey(0);

    }
    */
    flip_slices<<<no_blocks, MAX_THREADS>>>(sls_rs_d, N_sl * N_augment, w_rs, h_rs);

    cudaMemcpy(sls_rs, sls_rs_d, 2 * N_sl * N_augment * w_rs * h_rs * sizeof(float),
               cudaMemcpyDeviceToHost);

    delete [] sls_bx;
    delete [] Ls_bx;
    cudaFree(Ls_bx_d);
    cudaFree(sls_bx_d);
    cudaFree(sls_d);
    cudaFree(sls_rs_d);
}

void normalize_data(float *data, float *mean, float *std, unsigned *S)
{
    float *data_d;
    cudaMalloc((void **)&data_d, S[0] * S[1] * S[2] * S[3] * sizeof(float));
    cudaMemcpy(data_d, data, S[0] * S[1] * S[2] * S[3] * sizeof(float),
               cudaMemcpyHostToDevice);

    float *mean_d;
    cudaMalloc((void **)&mean_d,S[0] * S[1] * S[2] * sizeof(float));
    cudaMemcpy(mean_d, mean, S[0] * S[1] * S[2] * sizeof(float),
               cudaMemcpyHostToDevice);

    float *std_d;
    cudaMalloc((void **)&std_d,S[0] * S[1] * S[2] * sizeof(float));
    cudaMemcpy(std_d, std, S[0] * S[1] * S[2] * sizeof(float),
               cudaMemcpyHostToDevice);

    unsigned n_blcks = S[0] * S[1] * S[2] * S[3] / MAX_THREADS + 1;
    mean_std_normalization<<<n_blcks, MAX_THREADS>>>(data_d, mean_d, std_d,
                                                     S[0] * S[1] * S[2] * S[3],
                                                     S[0] * S[1] * S[2]);

    cudaFree(data_d);
    cudaFree(mean_d);
    cudaFree(std_d);
}
void extract_liver_side_ground_truth(unsigned char **masks_gt,
                                     unsigned *S, unsigned *Ls,
                                     unsigned N_s, unsigned *B, bool *gt)
{

    unsigned char *masks_gt_d;
    unsigned int *S_d;
    unsigned int *Ls_d;
    unsigned int *accs_d;
    unsigned int *accs;

    // 1.1 Allocate and transfer meta data to gpu
    cudaMalloc((void **)&masks_gt_d, Ls[N_s] * sizeof(unsigned char));
    unsigned int s = 0;
    for(unsigned int i = 0; i < N_s; i++)
    {
        cudaMemcpy(&masks_gt_d[Ls[i]], masks_gt[i],
                   (Ls[i+1] - Ls[i]) * sizeof(unsigned char),
                   cudaMemcpyHostToDevice);
        s += (S[3 * i] + S[3 * i + 1] + S[3 * i + 2]);
    }

    // 1.2 Allocate and transfer accumulators to gpu
    accs = new unsigned int[s];
    for(unsigned int i = 0; i < s; i++)
        accs[i] = 0;
    cudaMalloc((void **)&accs_d, s * sizeof(unsigned));
    cudaMemcpy(accs_d, accs, s * sizeof(unsigned), cudaMemcpyHostToDevice);

    // 1.3 Allocate and transfer sizes and lengths to gpu
    cudaMalloc((void **)&S_d, N_s * 3 * sizeof(unsigned));
    cudaMemcpy(S_d, S, N_s * 3 * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&Ls_d, (N_s + 1) * sizeof(unsigned));
    cudaMemcpy(Ls_d, Ls, (N_s + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);

    unsigned int grid_x = 512;
    unsigned int grid_y = Ls[N_s] / (grid_x * MAX_THREADS) + 1;
    dim3 grid(grid_x, grid_y);

    get_organ_mask_accs_gpu_multiple<<<grid, MAX_THREADS>>>(masks_gt_d,
                                                            Ls_d, S_d,
                                                            1, accs_d, N_s);
    cudaMemcpy(accs, accs_d, s * sizeof(unsigned), cudaMemcpyDeviceToHost);

    s = 0;
    for(unsigned int i = 0; i < N_s; i++)
    {
        unsigned h = (B[6 * i + 1] + B[6 * i]) / 2;
        float s_l = 0;
        float s_r = 0;
        for(unsigned j = 0; j < S[3 * i]; j++)
        {
            if(j <= h and accs[s + j])
                s_l += accs[s + j];
            else if(j > h and accs[s + j])
                s_r += accs[s + j];
        }
        s += (S[3 * i] + S[3 * i + 1] + S[3 * i + 2]);

        if((s_l / (h + 1)) > (s_r / (S[3 * i] -h - 1)))
            gt[i] = 1;
        else
            gt[i] = 0;
    }

    delete [] accs;
    cudaFree(Ls_d);
    cudaFree(S_d);
    cudaFree(accs_d);
    cudaFree(masks_gt_d);
}
