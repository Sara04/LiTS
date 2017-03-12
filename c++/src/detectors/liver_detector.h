/*
 * liver_detector.h
 *
 *  Created on: Mar 5, 2017
 *      Author: sara
 */

#ifndef LIVER_DETECTOR_H_
#define LIVER_DETECTOR_H_

#include "../lits_scan/lits_scan.h"
#include "liver_detector_cuda.cuh"


class LiTS_liver_detector
{
private:
    unsigned int *liver_bounding_box;

public:
    LiTS_liver_detector();
    void estimate_liver_bounding_box(LiTS_scan *scan);

};

#endif /* LIVER_DETECTOR_H_ */
