/*
 * liver_common.cuh
 *
 *  Created on: Jun 4, 2017
 *      Author: sara
 */

#ifndef LIVER_COMMON_METHODS_CUH_
#define LIVER_COMMON_METHODS_CUH_

/******************************************************************************
 * accumulate_for_mean_gpu: training data accumulation for mean computation
 *
 * Arguments:
 * 		input_data: array containing input data
 * 		acc_mean: accumulator for mean computation
 * 		S: size of the input data (height, width and number of samples)
 *****************************************************************************/
void accumulate_for_mean_gpu(float *input_data, float *acc_mean, unsigned *S);
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
		                    unsigned *S);
/******************************************************************************
 * normalize_data: mean and std normalization of input data
 *
 * Arguments:
 * 		data: input data to be normalized
 * 		mean: array containing data mean
 * 		std: array containing data std
 * 		S: data size (height, width and number of samples)
 *****************************************************************************/
void normalize_data(float *data, float *mean, float *std, unsigned *S);
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
		                      unsigned N_s);
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
		                     unsigned *Ls, unsigned N_s, unsigned int *accs);
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
                         unsigned N_s, unsigned *B);
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
                                     unsigned N_s, unsigned *B, bool *gt);
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
                           bool mirror);
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
                       bool mirror);
#endif /* LIVER_COMMON_METHODS_CUH_ */
