
#include "../segmentation/segmentator_cuda_kernels.cuh"
#include "liver_common_methods.cuh"

#include <iostream>

#define MAX_THREADS 1024

/******************************************************************************
 * accumulate_for_mean_gpu: training data accumulation for mean computation
 *
 * Arguments:
 * 		input_data: array containing input data
 * 		acc_mean: accumulator for mean computation
 * 		S: size of the input data (height, width and number of samples)
 *****************************************************************************/
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

/******************************************************************************
 * accumulate_for_std_gpu: training data accumulation for std computation
 *
 * Arguments:
 * 		input_data: array containing input data
 * 		acc_std: accumulator for mean computation
 * 		mean: input data mean
 * 		S: size of the input data (height, width and number of samples)
 *****************************************************************************/
void accumulate_for_std_gpu(float *input_data, float *acc_std, float *mean,
		                    unsigned *S)
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

/******************************************************************************
 * normalize_data: mean and std normalization of input data
 *
 * Arguments:
 * 		data: input data to be normalized
 * 		mean: array containing data mean
 * 		std: array containing data std
 * 		S: data size (height, width and number of samples)
 *****************************************************************************/
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

/******************************************************************************
 * determine_objects_bounds: determine bounds of binary objects whose
 * accumulations along each axes are provided in accumulator
 *
 * Arguments:
 * 		accs: accumulator containing objects' accumulations along each axes
 * 		S: sizes of each of the slice
 * 		B: array where boundaries would be saved
 * 		N_s: number of scans
 *****************************************************************************/
void determine_objects_bounds(unsigned *accs, unsigned *S, unsigned *B,
		                      unsigned N_s)
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

/******************************************************************************
 * organ_mask_accumulation: extraction of lungs' bounds
 *
 * Arguments:
 * 		masks_m: array of pointer to meta segmentation
 * 		S: sizes of meta segmentations
 * 		Ls: lengths of meta segmentations
 * 		N_s: number of scans
 * 		B: array where lung bounds would be stored
 *****************************************************************************/
void organ_mask_accumulation(unsigned char **masks_m, unsigned *S,
		                     unsigned *Ls, unsigned N_s, unsigned int *accs)
{
    unsigned char *masks_m_d;
    unsigned int *S_d;
    unsigned int *Ls_d;
    unsigned int *accs_d;

    // 1.1 Allocate and transfer meta segmentations to gpu
    cudaMalloc((void **)&masks_m_d, Ls[N_s] * sizeof(unsigned char));
    unsigned int s = 0;
    for(unsigned int i = 0; i < N_s; i++)
    {
        cudaMemcpy(&masks_m_d[Ls[i]], masks_m[i],
        		   (Ls[i+1] - Ls[i]) * sizeof(unsigned char),
                   cudaMemcpyHostToDevice);
        s += (S[3 * i] + S[3 * i + 1] + S[3 * i + 2]);
    }

    // 1.2 Allocate and transfer binary objects' accumulator
    accs = new unsigned int[s];
    for(unsigned int i = 0; i < s; i++)
        accs[i] = 0;
    cudaMalloc((void **)&accs_d, s * sizeof(unsigned));
    cudaMemcpy(accs_d, accs, s * sizeof(unsigned), cudaMemcpyHostToDevice);

    // 1.3 Allocate and transfer meta segmentation sizes and lengths to gpu
    cudaMalloc((void **)&S_d, N_s * 3 * sizeof(unsigned));
    cudaMemcpy(S_d, S, N_s * 3 * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&Ls_d, (N_s + 1) * sizeof(unsigned));
    cudaMemcpy(Ls_d, Ls, (N_s + 1) * sizeof(unsigned), cudaMemcpyHostToDevice);

    // 1.4 Accumulate binary masks of lungs along each axes
    unsigned int grid_x = 512;
    unsigned int grid_y = Ls[N_s] / (grid_x * MAX_THREADS) + 1;
    dim3 grid(grid_x, grid_y);
    get_organ_mask_accs_gpu_multiple<<<grid, MAX_THREADS>>>(masks_m_d,
                                                            Ls_d, S_d,
                                                            3, accs_d, N_s);

    // 1.5 Transfer accumulation back to cpu and determine lung bounds
    cudaMemcpy(accs, accs_d, s * sizeof(unsigned), cudaMemcpyDeviceToHost);

    cudaFree(Ls_d);
    cudaFree(S_d);
    cudaFree(accs_d);
    cudaFree(masks_m_d);
}

/******************************************************************************
 * extract_lung_bounds: extraction of lungs' bounds
 *
 * Arguments:
 * 		masks_m: array of pointer to meta segmentation
 * 		S: sizes of meta segmentations
 * 		Ls: lengths of meta segmentations
 * 		N_s: number of scans
 * 		B: array where lung bounds would be stored
 *****************************************************************************/
void extract_lung_bounds(unsigned char **masks_m, unsigned *S, unsigned *Ls,
                         unsigned N_s, unsigned *B)
{
	unsigned int *accs;
	organ_mask_accumulation(masks_m, S, Ls, N_s, accs);
    determine_objects_bounds(accs, S, B, N_s);
    delete [] accs;
}

/******************************************************************************
 * extract_liver_side_ground_truth: liver side ground truth extraction
 *
 * Arguments:
 * 		masks_gt: array of pointer to the ground truth segmentation
 * 		S: sizes of meta segmentations
 * 		Ls: lengths of meta segmentations
 * 		N_s: number of scans
 * 		B: array where lung bounds would be stored
 *		gt: array where ground truth would be stored
 *****************************************************************************/
void extract_liver_side_ground_truth(unsigned char **masks_gt,
                                     unsigned *S, unsigned *Ls,
                                     unsigned N_s, unsigned *B, bool *gt)
{

    unsigned int *accs;
    organ_mask_accumulation(masks_gt, S, Ls, N_s, accs);

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

        if((s_l / (h + 1)) > (s_r / (S[3 * i] - h - 1)))
            gt[i] = 1;
        else
            gt[i] = 0;
    }
    delete [] accs;
}

/******************************************************************************
 * extract_volume_slices: extraction of volume slices
 *
 * Arguments:
 * 		Vs: array of pointer to volumes
 * 		sls_rs: pointer to array where extracted resized slices would be stored
 * 		B: bounds of lungs' masks
 * 		N_s: number of scans/volumes
 * 		N_aug: augmentation factor
 * 		N_sl: number of slices
 * 		N_pix: number of pixels
 * 		w_rs: resized extracted width
 * 		h_rs: resized extracted height
 * 		ts_T: top slice to select
 * 		ts_B: bottom slice to select
 * 		random_rotate: random rotation for augmentation
 * 		bbox_shift: random bbox shifting for augmentation
 * 		mirror: weather to mirror data or not
 *****************************************************************************/
void extract_volume_slices(float **Vs, float *sls_rs,
		                   unsigned *B, unsigned *S,
		                   unsigned N_s, unsigned N_aug,
                           unsigned N_sl, unsigned N_pix,
                           unsigned w_rs, unsigned h_rs,
                           unsigned *ts_T, unsigned *ts_B,
                           float *random_rotate, unsigned *bbox_shift,
                           bool mirror)
{
    float *sls_d;
    float *sls_rs_d;
    unsigned int *Ls_bx_d;
    unsigned int *Ls_bx = new unsigned int[N_sl + 1];
    unsigned int *sls_bx = new unsigned int[4 * N_sl * N_aug];
    unsigned int *sls_bx_d;
    float *random_rotate_d;

    unsigned factor = 1;
    if(mirror)
    	factor = 2;
    cudaMalloc((void **)&sls_d, N_pix * sizeof(float));
    cudaMalloc((void **)&sls_rs_d,
    		   factor * N_sl * N_aug * w_rs * h_rs * sizeof(float));
    cudaMalloc((void **)&sls_bx_d, 4 * N_sl * N_aug * sizeof(unsigned int));
    cudaMalloc((void **)&Ls_bx_d, (N_sl + 1) * sizeof(unsigned int));
    cudaMalloc((void **)&random_rotate_d, 4 * N_sl * N_aug * sizeof(float));

    unsigned int inc = 0;
    Ls_bx[0] = 0;
    for(unsigned int s = 0; s < N_s; s ++)
        for(unsigned int i = ts_B[s]; i <= ts_T[s]; i++)
        {
            cudaMemcpy(&sls_d[Ls_bx[inc]], &Vs[s][i * S[3 * s] * S[3 * s + 1]],
                       S[3 * s] * S[3 * s + 1] * sizeof(float),
                       cudaMemcpyHostToDevice);
            Ls_bx[inc + 1] = Ls_bx[inc] + S[3 * s] * S[3 * s + 1];
            for(unsigned int a = 0; a < N_aug; a++)
                for(unsigned int j = 0; j < 4; j++)
                    sls_bx[4 * inc * N_aug + a * 4 + j] =
                    		B[6 * s + j] + bbox_shift[4 * s + j];
            inc += 1;
        }

    cudaMemcpy(sls_bx_d, sls_bx, 4 * N_sl * N_aug * sizeof(unsigned),
               cudaMemcpyHostToDevice);
    cudaMemcpy(random_rotate_d, random_rotate, 4 * N_sl * N_aug * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(Ls_bx_d, Ls_bx, (N_sl + 1) * sizeof(unsigned),
               cudaMemcpyHostToDevice);

    unsigned int no_blocks = (N_sl * N_aug * w_rs * h_rs) / MAX_THREADS + 1;

    resize_volume_slice_and_crop<<<no_blocks, MAX_THREADS>>>
    		(sls_d, sls_rs_d, sls_bx_d,  Ls_bx_d, random_rotate_d,
    		 N_sl, N_aug, w_rs, h_rs);

    if(mirror)
    	flip_slices<<<no_blocks, MAX_THREADS>>>
    		(sls_rs_d, N_sl * N_aug, w_rs, h_rs);

    cudaMemcpy(sls_rs, sls_rs_d,
    		   factor * N_sl * N_aug * w_rs * h_rs * sizeof(float),
               cudaMemcpyDeviceToHost);

    delete [] sls_bx;
    delete [] Ls_bx;
    cudaFree(Ls_bx_d);
    cudaFree(sls_bx_d);
    cudaFree(sls_d);
    cudaFree(sls_rs_d);
}

/******************************************************************************
 * extract_gt_slices: extraction of ground truth slices
 *
 * Arguments:
 * 		masks_gt: array of pointer to ground truth segmentation
 * 		sls_gt_rs: pointer to array where extracted resized gt would be stored
 * 		B: bounds of lungs' masks
 * 		N_s: number of scans/volumes
 * 		N_aug: augmentation factor
 * 		N_sl: number of slices
 * 		N_pix: number of pixels
 * 		w_rs: resized extracted width
 * 		h_rs: resized extracted height
 * 		ts_T: top slice to select
 * 		ts_B: bottom slice to select
 * 		random_rotate: random rotation for augmentation
 * 		bbox_shift: random bbox shifting for augmentation
 * 		mirror: wether to mirror data or not
 *****************************************************************************/
void extract_gt_slices(unsigned char **masks_gt, unsigned char *sls_gt_rs,
		               unsigned *B,  unsigned *S,
		               unsigned N_s, unsigned N_aug,
                       unsigned N_sl, unsigned N_pix,
                       unsigned w_rs, unsigned h_rs,
                       unsigned *ts_T, unsigned *ts_B,
                       float *random_rotate, unsigned *bbox_shift,
                       bool mirror)
{
    unsigned char *sls_gt_d;
    unsigned char *sls_gt_rs_d;
    unsigned int *Ls_bx = new unsigned int[N_sl + 1];
    unsigned int *Ls_bx_d;
    unsigned int *sls_bx = new unsigned int[4 * N_sl * N_aug];
    unsigned int *sls_bx_d;
    float *random_rotate_d;

    unsigned factor = 1;
    if(mirror)
    	factor = 2;

    cudaMalloc((void **)&sls_gt_d, N_pix * sizeof(unsigned char));
    cudaMalloc((void **)&sls_gt_rs_d,
    		   factor * N_sl * N_aug * w_rs * h_rs * sizeof(unsigned char));

    cudaMalloc((void **)&sls_bx_d, 4 * N_sl * N_aug * sizeof(unsigned int));
    cudaMalloc((void **)&Ls_bx_d, (N_sl + 1) * sizeof(unsigned int));
    cudaMalloc((void **)&random_rotate_d, 4 * N_sl * N_aug * sizeof(float));


    unsigned int inc = 0;
    Ls_bx[0] = 0;
    for(unsigned int s = 0; s < N_s; s ++)
    	for(unsigned int i = ts_B[s]; i <= ts_T[s]; i++)
        {
            cudaMemcpy(&sls_gt_d[Ls_bx[inc]],
            		   &masks_gt[s][i * S[3 * s] * S[3 * s + 1]],
                       S[3 * s] * S[3 * s + 1] * sizeof(unsigned char),
                       cudaMemcpyHostToDevice);
            Ls_bx[inc + 1] = Ls_bx[inc] + S[3 * s] * S[3 * s + 1];
            for(unsigned int a = 0; a < N_aug; a++)
                for(unsigned int j = 0; j < 4; j++)
                    sls_bx[4 * inc * N_aug + a * 4 + j] =\
                    	B[6 * s + j] + bbox_shift[4 * s + j];
            inc += 1;
        }

    cudaMemcpy(sls_bx_d, sls_bx, 4 * N_sl * N_aug * sizeof(unsigned),
               cudaMemcpyHostToDevice);
    cudaMemcpy(random_rotate_d, random_rotate, 4 * N_sl * N_aug * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(Ls_bx_d, Ls_bx, (N_sl + 1) * sizeof(unsigned),
               cudaMemcpyHostToDevice);

    unsigned int no_blocks = (N_sl * N_aug * w_rs * h_rs) / MAX_THREADS + 1;

    resize_gt_slice_and_crop<<<no_blocks, MAX_THREADS>>>
    		(sls_gt_d, sls_gt_rs_d, sls_bx_d, Ls_bx_d, random_rotate_d,
    	     N_sl, N_aug, w_rs, h_rs);

    cudaMemcpy(sls_gt_rs, sls_gt_rs_d,
    		   factor * N_sl * N_aug * w_rs * h_rs * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);

    delete [] sls_bx;
    delete [] Ls_bx;
    delete [] random_rotate;
    cudaFree(Ls_bx_d);
    cudaFree(sls_bx_d);
    cudaFree(sls_gt_d);
    cudaFree(sls_gt_rs_d);
    cudaFree(random_rotate_d);
}
