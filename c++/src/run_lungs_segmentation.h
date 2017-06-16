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

/******************************************************************************
 * This functions runs lungs segmentation of all training examples.
 * It is based on the voxel value thresholding, region growing and mathematical
 * morphology operations. Before the segmentation, voxel values are clipped and
 * normalized, and volume's axes are re-ordered and re-oriented to RAS or LAS
 * coordinate system.
 * !!! Note that the information provided in the scan's header about the axis
 * orientation that is normal to the sagittal plane is not reliable!!!
 * After the segmentation, axes of the meta-segmentation are re-ordered and
 * re-oriented to the original state.
 *****************************************************************************/
void run_train_lungs_segmentation(LiTS_db db);


#endif /* RUN_LUNGS_SEGMENTATION_H_ */
