/*
 * segmentation_tools.cu
 *
 *  Created on: Apr 15, 2017
 *      Author: sara
 */

#include "segmentation_tools.cuh"
#include "segmentator_cuda_kernels.cuh"

/******************************************************************************
 * compute_organ_mask_bounds: estimate left, right, front, back, top and bottom
 * lungs bounds based on extracted mask
 *
 * Arguments:
 *      meta_mask: meta mask containing organ segmentation
 *      value: value of the organ segmentation label
 *      left: left lungs bound
 *      right: right lungs bound
 *      front: front lungs bound
 *      back: back lungs bound
 *      top: top lungs bound
 *      bottom: bottom lungs bounds
 *
******************************************************************************/
void compute_organ_mask_bounds(const unsigned char *meta_mask,
                               const unsigned int *size,
                               const unsigned char value,
                               unsigned &left, unsigned &right,
                               unsigned &front, unsigned &back,
                               unsigned &bottom, unsigned &top)
{

    unsigned int n_p = size[0] * size[1] * size[2];
    unsigned int n_s = size[0] + size[1] + size[2];

    unsigned int n_p_B = n_p * sizeof(unsigned char);
    unsigned char *meta_mask_d;
    cudaMalloc((void **)&meta_mask_d, n_p_B);
    cudaMemcpy(meta_mask_d, meta_mask, n_p_B, cudaMemcpyHostToDevice);

    unsigned int n_s_B = n_s * sizeof(unsigned int);
    unsigned int *accs = new unsigned int[n_s];
    for(unsigned int i = 0; i < n_s; i++)
        accs[i] = 0;
    unsigned int *accs_d;
    cudaMalloc((void **)&accs_d, n_s_B);
    cudaMemcpy(accs_d, accs, n_s_B, cudaMemcpyHostToDevice);

    dim3 grid(size[1], size[2]);
    get_organ_mask_accs_gpu<<<grid, size[0]>>>(meta_mask_d, value, accs_d);

    cudaMemcpy(accs, accs_d, n_s_B, cudaMemcpyDeviceToHost);

    // 1. Determining left and right bounds
    for(unsigned int i = 0; i < size[0]; i++)
    {
        if(!left and accs[i])
            left = i;
        if(!right and accs[size[0] - i - 1])
            right = size[0] - i - 1;
        if(left and right)
            break;
    }

    // 2. Determining front and back bounds
    for(unsigned int i = 0; i < size[1]; i++)
    {
        if(!front and accs[size[0] + i])
            front = i;
        if(!back and accs[size[0] + size[1] - i - 1])
            back = size[1] - i - 1;
        if(front and back)
            break;
    }

    // 3. Determine bottom and top
    for(unsigned int i = 0; i < size[2]; i++)
    {
        if(!bottom and accs[size[0] + size[1] + i])
            bottom = i;
        if(!top and accs[size[0] + size[1] + size[2] - i - 1])
            top = size[2] - i - 1;
        if(bottom and top)
            break;
    }

    cudaFree(meta_mask_d);
    cudaFree(accs_d);
    delete [] accs;
}


