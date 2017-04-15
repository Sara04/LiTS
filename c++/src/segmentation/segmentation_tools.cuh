/*
 * segmentation_tools.cuh
 *
 *  Created on: Apr 15, 2017
 *      Author: sara
 */

#ifndef SEGMENTATION_TOOLS_CUH_
#define SEGMENTATION_TOOLS_CUH_

void compute_organ_mask_bounds(const unsigned char *meta_mask,
                               const unsigned int *size,
                               const unsigned char value,
                               unsigned &left, unsigned &right,
                               unsigned &front, unsigned &back,
                               unsigned &top, unsigned &bottom);



#endif /* SEGMENTATION_TOOLS_CUH_ */
