#include "tools.h"
#include <iostream>
#include <stdlib.h>

void region_growing_2d(bool *air_mask, const unsigned int *size,
                       unsigned int r_init, unsigned int c_init)
{
    unsigned int r_idx = r_init;
    unsigned int c_idx = c_init;
    int r_idx_n;
    int c_idx_n;

    bool *look_up = new bool[size[0] * size[1]];
    unsigned int *to_verify = new unsigned int[size[0] * size[1]];

    unsigned int to_verify_start = 0;
    unsigned int to_verify_end = 1;

    to_verify[0] = r_idx * size[0] + c_idx;
    for (unsigned int i = 0; i < size[0] * size[1]; i++)
        look_up[i] = false;
    look_up[to_verify[0]] = true;

    while (to_verify_start != to_verify_end)
    {
        if (air_mask[to_verify[to_verify_start]])
            air_mask[to_verify[to_verify_start]] = false;

        r_idx = to_verify[to_verify_start] / size[0];
        c_idx = to_verify[to_verify_start] - r_idx * size[0];
        to_verify_start += 1;

        for (int i_idx = -1; i_idx < 2; i_idx++)
        {
            for (int j_idx = -1; j_idx < 2; j_idx++)
            {
                if (i_idx != 0 or j_idx != 0)
                {
                    r_idx_n = (int) r_idx + i_idx;
                    c_idx_n = (int) c_idx + j_idx;

                    if (r_idx_n >= 0 and r_idx_n < size[1] and c_idx_n >= 0
                        and c_idx_n < size[0])
                    {
                        if ((!look_up[r_idx_n * size[0] + c_idx_n]) and
                            air_mask[r_idx_n * size[0] + c_idx_n])
                        {
                            to_verify[to_verify_end] =
                                    ((unsigned int) r_idx_n * size[0] +
                                     (unsigned int) c_idx_n);
                            look_up[to_verify[to_verify_end]] = true;
                            to_verify_end += 1;
                        }
                    }
                }
            }
        }
    }

    delete[] look_up;
    delete[] to_verify;
}

unsigned int region_growing_3d(bool *mask, const unsigned int *size,
                               unsigned int s_init, unsigned int r_init,
                               unsigned int c_init, unsigned int *labeled,
                               unsigned int label)
{
    unsigned int s_idx = s_init;
    unsigned int r_idx = r_init;
    unsigned int c_idx = c_init;
    int c_idx_n, r_idx_n, s_idx_n;

    bool *look_up = new bool[size[0] * size[1] * size[2]];
    unsigned int *to_verify = new unsigned int[size[0] * size[1] * size[2]];

    unsigned int to_verify_start = 0;
    unsigned int to_verify_end = 1;

    to_verify[0] = s_idx * size[0] * size[1] + r_idx * size[0] + c_idx;
    for (unsigned int i = 0; i < size[0] * size[1] * size[2]; i++)
        look_up[i] = false;
    look_up[to_verify[0]] = true;

    unsigned int object_size = 0;
    while (to_verify_start != to_verify_end)
    {
        if (mask[to_verify[to_verify_start]])
        {
            mask[to_verify[to_verify_start]] = false;
            labeled[to_verify[to_verify_start]] = label;
            object_size += 1;
        }

        s_idx = to_verify[to_verify_start] / (size[0] * size[1]);
        r_idx = (to_verify[to_verify_start] - s_idx * size[0] * size[1]) / size[0];
        c_idx = (to_verify[to_verify_start] - s_idx * size[0] * size[1]
                 - r_idx * size[0]);
        to_verify_start += 1;

        for (int k_idx = -1; k_idx < 2; k_idx++)
        {
            for (int i_idx = -1; i_idx < 2; i_idx++)
            {
                for (int j_idx = -1; j_idx < 2; j_idx++)
                {
                    if (i_idx != 0 or j_idx != 0)
                    {
                        s_idx_n = (int) s_idx + k_idx;
                        r_idx_n = (int) r_idx + i_idx;
                        c_idx_n = (int) c_idx + j_idx;

                        if (s_idx_n >= 0 and s_idx_n < size[2] and r_idx_n >= 0
                            and r_idx_n < size[1] and c_idx_n >= 0
                            and c_idx_n < size[0])
                        {
                            if ((!look_up[s_idx_n * size[0] * size[1]
                                    + r_idx_n * size[0] + c_idx_n])
                                and mask[s_idx_n * size[0] * size[1]
                                        + r_idx_n * size[0] + c_idx_n])
                            {
                                to_verify[to_verify_end] =
                                        ((unsigned int) s_idx_n * size[0]
                                         * size[1]
                                         + (unsigned int) r_idx_n * size[0]
                                         + (unsigned int) c_idx_n);
                                look_up[to_verify[to_verify_end]] = true;
                                to_verify_end += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    delete[] look_up;
    delete[] to_verify;
    return object_size;
}

void labeling_3d(const bool *mask, unsigned int *labeled,
                 const unsigned int *size, unsigned int *object_sizes,
                 unsigned int &label)
{
    bool *mask_copy = new bool[size[0] * size[1] * size[2]];
    for (unsigned int i = 0; i < (size[0] * size[1] * size[2]); i++)
    {
        mask_copy[i] = mask[i];
        labeled[i] = 0;
    }

    bool done = false;
    bool break_ = false;

    label = 1;

    while (!done)
    {
        for (unsigned int s_idx = 0; s_idx < size[2]; s_idx++)
        {
            for (unsigned int r_idx = 0; r_idx < size[1]; r_idx++)
            {
                for (unsigned int c_idx = 0; c_idx < size[0]; c_idx++)
                {
                    if (mask_copy[s_idx * size[0] * size[1] + r_idx * size[0]
                                  + c_idx])
                    {
                        object_sizes[label - 1] = region_growing_3d(mask_copy,
                                                                    size, s_idx,
                                                                    r_idx,
                                                                    c_idx,
                                                                    labeled,
                                                                    label);
                        break_ = true;
                        label += 1;
                    }
                    if (break_)
                        break;
                }
                if (break_)
                    break;
            }
            if (break_)
                break;
        }
        if (break_)
            break_ = false;
        else
            done = true;
    }
    delete[] mask_copy;
}

void center_of_mass(const unsigned int *labeled, const unsigned int *size,
                    unsigned int label, float &central_c, float &central_r,
                    float &central_s)
{
    unsigned int len = size[0] * size[1] * size[2];
    unsigned int count = 0;
    unsigned int c_idx, r_idx, s_idx;

    for (unsigned int i = 0; i < len; i++)
    {
        if (labeled[i] == label)
        {
            s_idx = i / (size[0] * size[1]);
            r_idx = (i - s_idx * size[0] * size[1]) / size[0];
            c_idx = i - s_idx * size[0] * size[1] - r_idx * size[0];

            central_s += float(s_idx) /size[2];
            central_r += float(r_idx) /size[1];
            central_c += float(c_idx) /size[0];
            count += 1;
        }
    }

    central_c /= count;
    central_r /= count;
    central_s /= count;
}

