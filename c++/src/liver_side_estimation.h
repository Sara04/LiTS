/*
 * liver_side_estimator.h
 *
 *  Created on: May 21, 2017
 *      Author: sara
 */

#ifndef LIVER_SIDE_ESTIMATION_H_
#define LIVER_SIDE_ESTIMATION_H_

#include <sys/time.h>

#include "lits_db/lits_database.h"
#include "lits_scan/lits_scan.h"
#include "lits_processing/lits_processor.h"
#include "liver_segmentation/liver_side_estimator.h"
#include "liver_segmentation/liver_side_estimator.cuh"

void save_liver_side_ground_truth(LiTS_db db);

void liver_side_estimator_train_and_valid(std::string model_path, LiTS_db db,
                                          unsigned N_iters=2000,
                                          unsigned N_subj_t=4,
                                          unsigned N_subj_v=10,
                                          unsigned N_augment_t=10,
                                          float lr=0.01);

void liver_side_estimator_eval(std::string model_path, LiTS_db db,
                               std::string scan_name);

void estimate_liver_side(std::string model_path, LiTS_db db,
                         std::string scan_name);

#endif /* LIVER_SIDE_ESTIMATION_H_ */
