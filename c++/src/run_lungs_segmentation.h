/*
 * run_lung_segmentation.h
 *
 *  Created on: May 21, 2017
 *      Author: sara
 */

#ifndef RUN_LUNGS_SEGMENTATION_H_
#define RUN_LUNGS_SEGMENTATION_H_

#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
namespace fs = boost::filesystem;
#include <sys/time.h>

#include "lits_db/lits_database.h"
#include "lits_scan/lits_scan.h"
#include "lits_processing/lits_processor.h"
#include "lung_segmentation/lung_segmentator.h"

void run_train_lungs_segmentation(LiTS_db db);


#endif /* RUN_LUNGS_SEGMENTATION_H_ */
