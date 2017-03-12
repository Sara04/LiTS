
#include "liver_detector.h"

LiTS_liver_detector::LiTS_liver_detector()
{
}

void LiTS_liver_detector::estimate_liver_bounding_box(LiTS_scan *scan)
{

    const unsigned int size[3] = {scan->get_width(),
                                  scan->get_height(),
                                  scan->get_depth()};

    const float voxel_size[3] = {scan->get_voxel_width(),
                                 scan->get_voxel_height(),
                                 scan->get_voxel_depth()};

    unsigned int *liver_bounds = new unsigned int[6];
    estimate_liver_lung_size(const_cast<bool *>(scan->get_lungs_mask()),
                             const_cast<unsigned char *>((scan->get_segmentation())->GetBufferPointer()),
                             size,
                             voxel_size,
                             const_cast<unsigned int *>((scan->get_body_bounds())),
                             liver_bounds);

    scan->set_liver_bbox(liver_bounds);

}
