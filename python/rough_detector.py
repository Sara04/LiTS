"""Class for liver bounding box detection."""
import numpy as np
from object_labeling import labeling
from object_labeling import region_growing
import copy


class LiTSroughDetector(object):
    """LiTSrough_detector class for body and organ detection.

    Attributes:

        air_threshold (float): threshold below whish everything is
                               considered as air, after value
                               normalization in range [-0.5, 0.5]
        min_lung_area_sa (float): minimal lung area in short-axis view
                                  in mm^2 that should be detectable
        min_body_area_sa (float): minimal body area in short-axis view
                                  in mm^2 that should be detectable
        lung_x_marge (float):  lung center marge along x axis
                               in detection bounding box
        lung_y_min (float): lung center min along y axis
                            in detection bounding box
        body_y_marge (float): body center marge along y axis
                              in a slice
        body_wall_width (float): minimal body wall width

        downsample_factor (int) - factor by which each slice
                                  is downsampled when detecting
                                  right lung

        extension_up (int): extension above liver along long axis
        extension_down (int): extension below liver along long axis
        lung_diff (int): difference between extensions for left and
                         right lung

    Methods:
        _body_bounds_sa: method for body bbox detection in one sa slice
        _region_size: method for object's area computation in mm^2
        _region_center: computes relative object's center in the given
                        binary image
        _right_lung_bottom: method for rough right lung bottom estimation
        _detect_right_lung: method for right lung bbox detection in sa view
                            in the slace at the position of estimated right
                            lung bottom
        detect_liver_bbox: method for liver bounding box estimation

    """

    def __init__(self, air_threshold=-0.49,
                 min_lung_area_sa=8000, min_body_area_sa=20000,
                 lung_x_marge=0.3, lung_y_min=0.3,
                 body_y_marge=0.4, body_wall_width=0.05,
                 downsample_factor=8,
                 extension_up=20, extension_down=200,
                 lung_diff=20):
        """Initialization method for LiTSroughDetector object."""
        self.air_threshold = air_threshold
        self.min_lung_area_sa = min_lung_area_sa
        self.min_body_area_sa = min_body_area_sa
        self.lung_x_marge = lung_x_marge
        self.lung_y_min = lung_y_min
        self.body_y_marge = body_y_marge
        self.body_wall_width = body_wall_width

        self.downsample_factor = downsample_factor
        self.extension_up = extension_up
        self.extension_down = extension_down
        self.lung_diff = lung_diff

    def _body_bounds_sa(self, slice_b, pixel_size):
        mask, label = labeling(np.copy(slice_b))
        for l in range(1, label + 1):
            candidate = (mask == l)
            s = self._region_size(candidate, pixel_size)
            if s < self.min_body_area_sa:
                continue
            c = self._region_center(candidate)
            if not (self.body_y_marge < c[0] < (1.0 - self.body_y_marge)):
                continue
            break

        y_s, y_e = np.where(np.sum(candidate, axis=1))[0][[0, -1]]
        x_s, x_e = np.where(np.sum(candidate, axis=0))[0][[0, -1]]
        return y_s, y_e, x_s, x_e, candidate

    def _region_size(self, region, pixel):

        return (np.sum(region) *
                pixel[0] * pixel[1] *
                self.downsample_factor * self.downsample_factor)

    def _region_center(self, region):

        h, w = region.shape
        rs = np.sum(region)

        on_x = np.sum(region, axis=0, dtype=float) / rs
        on_y = np.sum(region, axis=1, dtype=float) / rs

        on_x_c = np.sum(on_x * np.arange(0, w), dtype=float) / w
        on_y_c = np.sum(on_y * np.arange(0, h), dtype=float) / h

        return on_y_c, on_x_c

    def _right_lung_bottom(self, mask_p, mask_f):

        p_s, p_e = np.where(np.sum(mask_p, axis=1))[0][[0, -1]]
        f_s, f_e = np.where(np.sum(mask_f, axis=1))[0][[0, -1]]

        p_lb = np.where(np.sum(mask_p, axis=0) > (p_e - p_s) / 2)[0][0]
        f_lb = np.where(np.sum(mask_f, axis=0) > (f_e - f_s) / 2)[0][0]

        lb = max([p_lb, f_lb])

        return lb

    def _detect_right_lung(self, scan):

        lung_detected = False

        scan_b = scan.volume[::self.downsample_factor,
                             ::self.downsample_factor,
                             :] > self.air_threshold

        for i in range(1, scan.d):

            slice_b = scan_b[:, :, scan.d - i]

            y_s, y_e, x_s, x_e, candidate =\
                self._body_bounds_sa(slice_b, scan.voxel_size[0:2])
            y_c = (y_s + y_e) / 2

            right_lung = (candidate[y_s:y_c, x_s:x_e] == 0)
            mask, label = labeling(np.copy(right_lung))

            for l in range(1, label + 1):

                candidate = (mask == l)
                s = self._region_size(candidate, scan.voxel_size[0:2])
                c = self._region_center(candidate)

                if s < self.min_lung_area_sa:
                    continue

                if not (self.lung_x_marge < c[1] < (1. - self.lung_x_marge) and
                        c[0] > self.lung_y_min):
                    continue

                c_s, c_e = np.where(np.sum(candidate, axis=0))[0][[0, -1]]

                if (c_s < self.body_wall_width * candidate.shape[1] or
                   c_e > (1. - self.body_wall_width) * candidate.shape[1]):
                    continue

                lung_detected = True

                init_indices = []
                for idx in np.where(candidate[c[0] * (y_c - y_s), :])[0]:
                    init_indices.append([idx, scan.d - i])
                slice_p = scan_b[y_s + c[0] * (y_c - y_s), x_s:x_e, :] == 0
                mask_p = region_growing(init_indices, np.copy(slice_p))

                init_indices = []
                for idx in np.where(candidate[:, c[1] * (x_e - x_s)])[0]:
                    init_indices.append([idx, scan.d - i])
                slice_f = scan_b[y_s:y_c, x_s + c[1] * (x_e - x_s), :] == 0
                mask_f = region_growing(init_indices, np.copy(slice_f))

            if lung_detected:
                break

        if not lung_detected:
            return False, 0.0
        else:
            lb = self._right_lung_bottom(mask_p, mask_f)
            y_s, y_e, x_s, x_e, _ =\
                self._body_bounds_sa(scan_b[:, :, lb], scan.voxel_size[0:2])
            return True, [x_s, x_e, y_s, y_e, lb]

    def detect_liver_bbox(self, scan):
        """Detect liver bounding box.

        The method updates input scan object with the scan in the detected
        bounding box and accordingly other attributes of the scan object.

        Firstly, a method for right lung detection is performed on the original
        scan and if the lung is not detected scan is mirrored and the approach
        is performed again.

        From the right lung's estimated position (or the left lung's if the
        right one is not detectable) a liver bounding box is estimated.

        Args:
            scan (LiTSscan): object containing volume, segmentation and info
        """
        is_lung_detected, bounds = self._detect_right_lung(scan)

        extension_up = self.extension_up
        extension_down = self.extension_down
        if not is_lung_detected:
            scan_mirror = copy.copy(scan)
            scan_mirror.volume = np.copy(scan.volume)[::-1, :, :]

            is_lung_detected, bounds = self._detect_right_lung(scan_mirror)

            extension_up += self.lung_diff
            extension_down -= self.lung_diff

        if is_lung_detected:
            back, front, right, left =\
                [bounds[i] * self.downsample_factor for i in range(4)]
            lb = bounds[4]

            top = min([scan.d - 1,
                       lb + int(extension_up / scan.voxel_size[2])])
            bottom = max([0,
                          lb - int(extension_down / scan.voxel_size[2])])

            scan.segmentation =\
                scan.segmentation[right:left, back:front, bottom:top]
            scan.volume = scan.volume[right:left, back:front, bottom:top]
            scan.h, scan.w, scan.d = scan.volume.shape
