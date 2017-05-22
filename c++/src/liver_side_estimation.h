/*
 * liver_side_estimator.h
 *
 *  Created on: May 21, 2017
 *      Author: sara
 */

#ifndef LIVER_SIDE_ESTIMATION_H_
#define LIVER_SIDE_ESTIMATION_H_

#include <boost/filesystem.hpp>
#include <boost/progress.hpp>
namespace fs = boost::filesystem;
#include <sys/time.h>

#include "lits_db/lits_database.h"
#include "lits_scan/lits_scan.h"
#include "lits_processing/lits_processor.h"
#include "liver_segmentation/liver_side_estimator.h"

void liver_side_estimator_train_and_valid(LiTS_db db);



#endif /* LIVER_SIDE_ESTIMATION_H_ */
