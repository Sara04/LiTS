/*
 * liver_side_estimator.cuh
 *
 *  Created on: Apr 16, 2017
 *      Author: sara
 */

#ifndef LIVER_SIDE_ESTIMATOR_CUH_
#define LIVER_SIDE_ESTIMATOR_CUH_

#include <cstdlib>
#include <iostream>

void extract_lung_bounds(unsigned char **masks_m, unsigned *S,
                         unsigned *Ls, unsigned N_s, unsigned *B);

void extract_slices(float **Vs, float *sls_rs, unsigned *B, unsigned *S,
                    unsigned N_s, unsigned N_augment,
                    unsigned N_sl, unsigned N_pix,
                    unsigned w_rs, unsigned h_rs,
                    unsigned *ts_T, unsigned *ts_B,
                    unsigned max_shift);

void extract_liver_side_ground_truth(unsigned char **masks_gt,
                                     unsigned *S, unsigned *Ls,
                                     unsigned N_samples,
                                     unsigned *B, bool *gt);

void determine_bounds(unsigned *accs, unsigned *S, unsigned *B, unsigned N_s);

void accumulate_for_mean_gpu(float *input_data, float *mean, unsigned *S);

void accumulate_for_std_gpu(float *input_data, float *std, float *mean, unsigned *S);

void normalize_data(float *data, float *mean, float *std, unsigned *S);

#endif /* LIVER_SIDE_ESTIMATOR_CUH_ */
