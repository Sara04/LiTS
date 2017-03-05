/*
 * tools.h
 *
 *  Created on: Feb 25, 2017
 *      Author: sara
 */

#ifndef TOOLS_H_
#define TOOLS_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void region_growing_2d(bool *air_mask, const unsigned int *size,
                       unsigned int r_init, unsigned int c_init);

unsigned int region_growing_3d(bool *mask, const unsigned int *size,
                               unsigned int s_init, unsigned int r_init,
                               unsigned int c_init, unsigned int *labeled,
                               unsigned int label);

void labeling_3d(const bool *mask, unsigned int *labeled,
                 const unsigned int *size, unsigned int *object_sizes,
                 unsigned int &label);

void center_of_mass(const unsigned int *labeled, const unsigned int *size,
                    unsigned int label, float &central_c, float &central_r,
                    float &central_s);

#endif /* TOOLS_H_ */
