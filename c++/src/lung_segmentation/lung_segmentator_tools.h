/*
 * lung_segmentator_tools.h
 *
 *  Created on: Apr 8, 2017
 *      Author: sara
 */

#ifndef LUNG_SEGMENTATOR_TOOLS_H_
#define LUNG_SEGMENTATOR_TOOLS_H_

#include "../tools/tools.h"

bool is_in_body_box(const unsigned int *slice_bounds, unsigned int c_idx,
                    unsigned int r_idx);

void remove_outside_body_air(bool *air_mask, const unsigned int *size,
                             const unsigned int *bounds);

void extract_lung_labels(const unsigned int *labeled, bool *candidates,
                         const unsigned int *size,
                         const unsigned int *main_labels, unsigned int count);

void extract_lung_candidates(const unsigned int *labeled,
                             const unsigned int *size,
                             unsigned int *object_sizes, unsigned int &label,
                             bool *candidates, float &size_threshold,
                             const float *lung_assumed_c_n,
                             unsigned ng_f=10, unsigned lr_f=2,
                             float c_th=0.4, float r_th=0.3, float s_th=0.6,
                             float r_d=0.15, float s_d=0.15);

void lung_central_slice(bool *air_mask, unsigned int * volume_s,
                        float &lung_center_slice);

#endif /* LUNG_SEGMENTATOR_TOOLS_H_ */
