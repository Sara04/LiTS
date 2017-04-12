/*
 * lung_segmentator_tools.cpp
 *
 *  Created on: Apr 8, 2017
 *      Author: sara
 */

#include <fstream>
#include <iostream>
#include "lung_segmentator_tools.h"

/******************************************************************************
 * is_in_body_box: verify if pixel coordinates are within bounds
 *
 * Arguments:
 *      bbox: side, front and back bounds
 *      r: pixel's row
 *      c: pixel's column
******************************************************************************/
bool is_in_body_box(const unsigned *bbox, unsigned c, unsigned r)
{
    if (bbox[0] <= c and c <= bbox[1] and bbox[2] <= r and r <= bbox[3])
        return true;
    return false;
}

/******************************************************************************
 * remove_outside_body_air: remove air segments that are outside
 *      body, and for sure do not belong to the lungs
 *
 * Arguments:
 *      air: air mask segmentation
 *      S: air mask size
 *      bboxes: body bounding box for all slices
******************************************************************************/
void remove_outside_body_air(bool *air, const unsigned int *S,
                             const unsigned int *bboxes)
{
    unsigned int bs[4];
    for (unsigned int s = 0; s < S[2]; s++)
    {
        for (unsigned i = 0; i < 4; i++)
            bs[i] = bboxes[s + i * S[2]];

        for (unsigned int r = 0; r < S[1]; r++)
        {
            for (unsigned int c = 0; c < S[0]; c++)
            {
                if (!is_in_body_box(bs, c, r))
                    air[s * S[0] * S[1] + r * S[0] + c] = false;
                else
                {
                    if ((r == bs[2] or r == bs[3]) and c > bs[0] and c < bs[1])
                        if (air[s * S[0] * S[1] + r * S[0] + c])
                            region_growing_2d(&air[s * S[0] * S[1]], S, r, c);

                    if ((c == bs[0] or c == bs[1]) and r > bs[2] and r < bs[3])
                        if (air[s * S[0] * S[1] + r * S[0] + c])
                            region_growing_2d(&air[s * S[0] * S[1]], S, r, c);
                }
            }
        }
    }
}

/******************************************************************************
 * lung_central_slice: estimate lung central slice and remove all candidates
 * that are far from this estimated central slice
 *
 * Arguments:
 *      air: air segmentation
 *      S: size of the air mask
 *      lungs_c_s: central lungs' slice
******************************************************************************/
void lung_central_slice(bool *air, unsigned int * S, float &lungs_c_s)
{
    // 1. Average air area over slice
    float *coronal_sum = new float[S[2]];
    float c_max = 0;
    unsigned int c_max_idx = 0;
    for(unsigned int s = 0; s < S[2]; s++)
    {
        coronal_sum[s] = 0;
        for(unsigned int r = 0; r < S[1]; r++)
            for(unsigned int c = 0; c < S[0]; c++)
                coronal_sum[s] += r * air[s * S[0] * S[1] + r * S[0] + c];
        coronal_sum[s] /= (S[0] * S[1]);
        if (coronal_sum[s] > c_max)
        {
            c_max = coronal_sum[s];
            c_max_idx = s;
        }
    }
    float threshold = 2.0;
    unsigned int th_idx_low = 0;
    unsigned int th_idx_high = S[2] - 1;
    if(c_max < 10.0)
        threshold = 1.0;
    // 2. Lower bound
    for(unsigned int s = c_max_idx; s >= 0; s--)
        if (coronal_sum[s] <= threshold)
        {
            th_idx_low = s;
            break;
        }
    // 3. Upper bound
    for(unsigned int s = c_max_idx; s < S[2]; s++)
        if (coronal_sum[s] <= threshold)
        {
            th_idx_high = s;
            break;
        }
    // 4. Find mass center
    float lung_c = 0;
    float lung_sum = 0;
    for(unsigned int s = th_idx_low; s < th_idx_high; s++)
    {
        lung_c += (s * coronal_sum[s]);
        lung_sum += coronal_sum[s];
    }
    lungs_c_s = lung_c / lung_sum;
    int lungs_bottom = int(th_idx_low - 0.2 * (lungs_c_s - th_idx_low));
    // 5. Remove all air objects below bottom lungs bound
    for(unsigned int s = 0; s < lungs_bottom; s++)
        for(unsigned int i = 0; i < (S[0] * S[1]); i++)
                air[s * S[0] * S[1] + i] = 0.0;
}

/******************************************************************************
 * extract_lung_labels: extract segmented objects from the array labeled
 *      for the first count labels from the main_labels array
 *
 * Arguments:
 *      labeled: pointer to the array containing labeled within body air
 *          segments
 *      candidates: pointer to the array where segments with desired
 *          labels would be placed
 *      size: pointer to the array containing size of the labeled array
 *      main_labels: pointer to the array containing object labels
 *      count: number of labels from the array main_labels to be
 *          extracted from the labeled array
******************************************************************************/
void extract_lung_labels(const unsigned int *labeled, bool *candidates,
                         const unsigned int *size,
                         const unsigned int *main_labels, unsigned int count)
{
    unsigned int len = size[0] * size[1] * size[2];
    for (unsigned int i = 0; i < len; i++)
    {
        for (unsigned int j = 0; j < count; j++)
        {
            if (labeled[i] == main_labels[j])
            {
                candidates[i] = true;
                break;
            }
            candidates[i] = false;
        }
    }
}

/******************************************************************************
 * extract_lung_candidates: extract binary mask that covers both lung wings
 *
 * Arguments:
 *      labeled: pointer to the array containing labeled within body
 *          air segments
 *      size: pointer to the array containing size of the labeled array
 *      object_sizes: pointer to the array where the size of the segmented
 *          objects are placed
 *      label: reference to the number of labels in the labeled array
 *      candidates: pointer to the array where the mask corresponding to the
 *          lung wings would be placed
 *      size_threshold: reference to the estimated lung size threshold
 *      lung_assumed_c_n: assumed normalized center of the lungs
 *      ng_f: negligible factor
 *      lr_f: lung ratio factor
 *      c_th: normalized column center threshold
 *      r_th: normalized row center threshold
 *      s_th: normalized slice threshold
 *      r_d: normalized row center distance threshold
 *      s_d: normalized slice center distance threshold
******************************************************************************/
void extract_lung_candidates(const unsigned int *labeled,
                             const unsigned int *S,
                             unsigned int *object_sizes, unsigned int &label,
                             bool *candidates, float &size_threshold,
                             const float *lung_assumed_c_n,
                             unsigned ng_f, unsigned lr_f,
                             float c_th, float r_th, float s_th,
                             float r_d, float s_d)
{
    // 1. Count the number of labeled segments whose size is not negligible
    unsigned int count = 0;
    for (unsigned int i = 1; i < label; i++)
        count += (object_sizes[i - 1] > (size_threshold / ng_f));
    // 2. Sort main segment candidates according to their size
    unsigned int *MC = new unsigned int[count];
    unsigned int *ML = new unsigned int[count];
    unsigned int p;
    count = 0;
    for (unsigned int i = 1; i < label; i++)
        if (object_sizes[i - 1] > (size_threshold / ng_f))
        {
            p = count;
            for (unsigned int j = 0; j < count; j++)
            {
                if (object_sizes[i - 1] > MC[j])
                {
                    for (unsigned int k = count; k > j; k--)
                    {
                        MC[k] = MC[k - 1];
                        ML[k] = ML[k - 1];
                    }
                    p = j;
                    break;
                }
            }
            MC[p] = object_sizes[i - 1];
            ML[p] = i;
            count += 1;
        }
    // 3.1. If there is only one object, consider it as the mask object
    //      belonging to the both lungs
    if (count == 1)
    {
        extract_lung_labels(labeled, candidates, S, ML, 1);
        goto end_detection1;
    }
    // 3.2. In addition to the segment size, include information about
    //      segment position into decision making
    else
    {
        // 3.2.1. Determine normalized centers of mass for each segment whose
        //        size is not negligible
        float *c = new float[count];
        float *r = new float[count];
        float *s = new float[count];
        for (unsigned int i = 0; i < count; i++)
        {
            c[i] = 0.0;
            r[i] = 0.0;
            s[i] = 0.0;
            center_of_mass(labeled, S, ML[i], c[i], r[i], s[i]);
        }
        // 3.2.2. If the largest segment is well centered and larger enough
        //        then the second largest segment, it is considered as the
        //        segment belonging to the both lung wings

        if (c[0] < (1. - c_th) and c[0] > c_th and r[0] < (1. - r_th) and
            r[0] > r_th and s[0] > s_th and (MC[0] / MC[1]) > lr_f)
        {
            extract_lung_labels(labeled, candidates, S, ML, 1);
            goto end_detection2;
        }
        // 3.2.3. Verify if there is a pair of segments covering one each
        //        lung separately, or if that is not case find the segment
        //        that is closest to the assumed center
        else
        {
            // 3.2.3.1 Verify if there is a pair of segments that most likely
            //         correspond to the lung wings
            for (unsigned int i = 0; i < (count - 1); i++)
                for (unsigned int j = i + 1; j < count; j++)
                    if (((c[i] < c_th and c[j] > (1. - c_th)) or
                        (c[j] < c_th and c[i] > (1. - c_th))) and
                        abs(r[i] - r[j]) < r_d and abs(s[i] - s[j]) < s_d)
                    {
                        ML[0] = ML[i];
                        ML[1] = ML[j];
                        extract_lung_labels(labeled, candidates, S, ML, 2);
                        goto end_detection2;
                    }
            // 3.2.3.2 Find the segment that is closest to the assumed center
            float min_dist = 1.0;
            float current_dist = 0.0;
            unsigned int m = 0;
            for(unsigned int i = 0; i < count; i++)
            {
                current_dist = sqrt(
                        pow(c[i] - lung_assumed_c_n[0], 2) +
                        pow(r[i] - lung_assumed_c_n[1], 2) +
                        pow(s[i] - lung_assumed_c_n[2], 2));
                if (current_dist < min_dist)
                {
                    min_dist = current_dist;
                    m = i;
                }
            }
            ML[0] = ML[m];
            extract_lung_labels(labeled, candidates, S, ML, 1);
            goto end_detection2;
        }
        // 3.2.4. Release memory from arrays containing position info
        end_detection2:
        {
            delete[] c;
            delete[] r;
            delete[] s;
            goto end_detection1;
        }
    }
    // 4. Release memory containing size and label info
    end_detection1:
    {
        delete[] MC;
        delete[] ML;
        return;
    }
}


