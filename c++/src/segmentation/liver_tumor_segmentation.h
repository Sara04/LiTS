/*
 * liver_tumor_segmentation.h
 *
 *  Created on: Mar 12, 2017
 *      Author: sara
 */

#ifndef LIVER_TUMOR_SEGMENTATION_H_
#define LIVER_TUMOR_SEGMENTATION_H_

#include "../lits_scan/lits_scan.h"
#include "liver_tumor_segment_cuda.cuh"
#include <random>

class LiTS_segmentator
{
private:

    // low level features weights
    double *W_lf;
    double *b_lf;

public:

    LiTS_segmentator();
    ~LiTS_segmentator();
    void development(std::list<LiTS_scan> development_set);
    void validation(std::list<LiTS_scan> validation_set);
    void evaluation(std::list<LiTS_scan> evaluation_set);
};



#endif /* LIVER_TUMOR_SEGMENTATION_H_ */
