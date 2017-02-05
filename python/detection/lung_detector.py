"""Class for tumor detection training, validation and testing."""
import numpy as np
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from tools.object_labeling import region_growing, labeling_3d, opening_3d
from tools.stat_functions import volume_center_3d


class LiTSlungDetector(object):
    """LiTSliverDetector class for liver segmentation.

    Attributes:
        downsampling_factor (int) - factor by which slices are
            downsampled for the analysis
        air_threshold (float) - threshold below which everything
            is considered as air
        min_lung_volume (float) - minimal lung volume used to decrease
            the number of lung candidates (mm^3)
        lungs (numpy array) - 3d binary array containing lung detection
    Methods:
        _body_bounds - determining body bounds in 2d binary image
            (top, bottom, left and right bounds of the body)
        _remove_outside_body - remove air areas that are outside the body
            (set their value to 1, so that only air inside the body has
             value 0)
        _detect_lungs - determining lungs out of all air volumes in the body
        lung_detection - calling functions for body bounds detection, removal
            of outside of the body and lung detection
    """

    def __init__(self, downsampling_factor=8, air_threshold=-0.49,
                 min_lung_volume=10000.0):
        """Initialization method for LiTSliverDetector object.

        Args:
            downsampling_factor (int) - factor by which slices are
                downsampled for the analysis
            air_threshold (float) - threshold below which everything
                is considered as air
            min_lung_volume (float) - minimal lung volume used to decrease
                the number of lung candidates (mm^3)

        """
        self.downsampling_factor = downsampling_factor
        self.air_threshold = air_threshold
        self.min_lung_volume = min_lung_volume
        self.lungs = None

    def _body_bounds(self, slice):

        h, w = slice.shape

        slice_sum_v = np.sum(slice, axis=0)
        slice_sum_h = np.sum(slice, axis=1)

        left_bound, right_bound, top_bound, bottom_bound = [0, w - 1, 0, h - 1]

        x = np.where(slice_sum_v[0:w / 2] == 0)[0]
        if len(x):
            left_bound = np.min([x[-1] + 1, w / 2 - 1])

        x = np.where(slice_sum_v[w / 2:] == 0)[0]
        if len(x):
            right_bound = np.max([x[0] - 1 + w / 2, w / 2])

        x = np.where(slice_sum_h[0:h / 2] == 0)[0]
        if len(x):
            top_bound = np.min([x[-1] + 1, h / 2 - 1])

        x = np.where(slice_sum_h[h / 2:] == 0)[0]
        if len(x):
            bottom_bound = np.max([x[0] - 1 + h / 2, h / 2])

        return [top_bound, bottom_bound, left_bound, right_bound]

    def _remove_outside_body(self, img_patch):

        h, w = img_patch.shape

        se = generate_binary_structure(2, 2)
        img_patch = binary_dilation(img_patch, structure=se).astype('uint8')

        for c in [0, w - 1]:
            x = np.where(img_patch[:, c] == 0)[0]
            while(len(x)):
                mask = region_growing([[x[0], c]], img_patch == 0)
                img_patch += mask
                x = np.where(img_patch[:, c] == 0)[0]

        for r in [0, h - 1]:
            x = np.where(img_patch[r, :] == 0)[0]
            while(len(x)):
                mask = region_growing([[r, x[0]]], img_patch == 0)
                img_patch += mask
                x = np.where(img_patch[r, :] == 0)[0]

        img_patch = binary_erosion(img_patch, structure=se).astype('uint8')

        return img_patch

    def _detect_lungs(self, candidates, voxel_size,
                      z_min=0.6, lung_intest=50, x_to_c=0.5):

        # should be done smarter
        h, w, d = candidates.shape

        candidates = opening_3d(candidates)
        mask, label = labeling_3d(candidates)

        labeled_candidates = []
        for l in range(1, label + 1):
            volume = (np.sum(mask == l) *
                      self.downsampling_factor *
                      self.downsampling_factor *
                      voxel_size[0] *
                      voxel_size[1] *
                      voxel_size[2])
            if volume < self.min_lung_volume:
                continue
            labeled_candidates.append([volume, l])

        labeled_sorted = sorted(labeled_candidates, reverse=True)

        lungs_found = False
        for i in range(len(labeled_sorted)):
            if lungs_found:
                break
            object_1 = (mask == labeled_sorted[i][1])
            c_y_1, c_x_1, c_z_1 = volume_center_3d(object_1)
            if c_z_1 < z_min:
                continue
            for j in range(len(labeled_sorted)):
                if j != i:
                    object_2 = (mask == labeled_sorted[j][1])
                    c_y_2, c_x_2, c_z_2 = volume_center_3d(object_2)

                    d_z = (c_z_1 - c_z_2) * voxel_size[2] * d
                    d_y = (c_y_1 - c_y_2) * voxel_size[1] * h
                    d_y_z = np.sqrt(d_y ** 2 + d_z ** 2)

                    if d_y_z > lung_intest and np.abs(c_x_1 - 0.5) < x_to_c:
                        lungs = object_1
                        lungs_found = True
                        break
                    else:
                        if abs(d_y_z) < lung_intest:
                            lungs = object_1 + object_2
                            lungs_found = True
                            break
        if not lungs_found:
            lungs = (mask == labeled_sorted[0][1])
        return lungs

    def lung_detection(self, scan):
        """Detect lungs.

        Args:
            scan (LiTSscan) - scan object
        """
        volume_ds = scan.volume[::self.downsampling_factor,
                                ::self.downsampling_factor,
                                :]
        h, w, d = volume_ds.shape

        candidates = np.zeros((h, w, d), dtype='uint8')

        for i in range(scan.d - 1, 0, -1):

            slice_ds = (volume_ds[:, :, i] > self.air_threshold).\
                astype('uint8')

            body_bounds = self._body_bounds(slice_ds)
            img_patch = slice_ds[body_bounds[0]:body_bounds[1],
                                 body_bounds[2]:body_bounds[3]]

            img_patch = self._remove_outside_body(img_patch)
            candidates[body_bounds[0]:body_bounds[1],
                       body_bounds[2]:body_bounds[3], i] = (img_patch == 0)

        self.lungs = self._detect_lungs(candidates, scan.voxel_size)
