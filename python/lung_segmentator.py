"""LiTSLungSegmentator class."""
import numpy as np
from scipy.ndimage import measurements


class LiTSLungSegmentator(object):
    """LiTSLungSegmentator class for lungs segmentation.

    Attributes:
        ds_f - downsampling factors along each axis
            (in order to speed up segmentation)
        lv_th - minimal assumed lung size
        air_th - threshold below which voxels are considered to belong
            to air regions
        lung_assumed_center_n - assumed normalized lungs' center position
        body_bounds_th - thresholds for determining body bounds per slice
        cth - assumed normalized lungs center in left-right direction
        rth - assumed normalized lungs center in front-back direction
        sth - assumed normalized lungs center in bottom-top direction
        lr_f - air object negligence factor
        r_d - allowed normalized difference between lungs centers
            along front-back axis
        s_d - allowed normalized difference between lungs centers
            along bottom-top axis

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

    def __init__(self, ds_f=[4, 4, 1], lv_th=50000,
                 air_th=-0.49, lung_assumed_center_n=[0.5, 0.6, 0.9],
                 body_bounds_th=[0.0003, 0.0003, 0.0007],
                 cth=0.4, rth=0.3, sth=0.6, lr_f=2, r_d=0.15, s_d=0.15):
        """Initialization method for LiTSLungSegmentator object.

        Arguments:
            ds_f - downsampling factors along each axis
                (in order to speed up segmentation)
            lv_th - minimal assumed lung size
            air_threshold - threshold below which voxels are considered to
                belong to air region in voxels
            lung_assumed_center_n - assumed normalized lungs' center position
            body_bounds_th - thresholds for determining body bounds per slice
            cth - assumed normalized lungs center in left-right direction
            rth - assumed normalized lungs center in front-back direction
            sth - assumed normalized lungs center in bottom-top direction
            lr_f - air object negligence factor
            r_d - allowed normalized difference between lungs centers
                along front-back axis
            s_d - allowed normalized difference between lungs centers
                along bottom-top axis
        """
        self.ds_f = ds_f
        self.lv_th = lv_th
        self.air_th = air_th
        self.lung_assumed_center_n = lung_assumed_center_n
        self.body_bounds_th = body_bounds_th

        self.cth = cth
        self.rth = rth
        self.sth = sth

        self.lr_f = lr_f
        self.r_d = r_d
        self.s_d = s_d

    def _extract_lung_candidates(self, count_sgnf, labels_list, masks,
                                 object_sizes):

        h, w, d = masks.shape
        obj_idx_s = np.argsort(object_sizes)[::-1]
        obj_s_s = [object_sizes[idx] for idx in obj_idx_s]
        labels_s = [labels_list[idx] for idx in obj_idx_s]

        # 7.1 If there is only one significant object that one corresponds to
        # the lungs
        lungs_segmented = False
        if count_sgnf == 1:
            lungs = (masks == labels_s[0])
            lungs_segmented = True
        else:
            pos = np.zeros((count_sgnf, 3))
            for i in range(count_sgnf):
                obj = (masks == labels_s[i])
                obj_s_c = np.sum(obj, axis=2)
                o_h = np.sum(obj_s_c, axis=0).astype('float32') / obj_s_s[i]
                o_v = np.sum(obj_s_c, axis=1).astype('float32') / obj_s_s[i]
                o_d = np.sum(np.sum(obj, axis=0), axis=0).astype('float32') /\
                    obj_s_s[i]

                pos[i, 0] = np.sum(o_h * np.arange(w).astype('float32')) / w
                pos[i, 1] = np.sum(o_v * np.arange(h).astype('float32')) / h
                pos[i, 2] = np.sum(o_d * np.arange(d).astype('float32')) / d

            # 7.2.1. If the largest segment is well centered and larger enough
            #        then the second largest segment, it is considered as the
            #        segment belonging to the both lung wings
            if(pos[0, 0] < 1. - self.cth and pos[0, 0] > self.cth and
               pos[0, 1] < 1. - self.rth and pos[0, 1] > self.rth and
               pos[0, 2] > self.sth and obj_s_s[0] / obj_s_s[1] > self.lr_f):
                lungs = (masks == labels_s[0])
                lungs_segmented = True
            else:
                for i in range(count_sgnf - 1):
                    for j in range(i, count_sgnf):
                        if((pos[i, 0] < self.cth and
                            pos[j, 0] > 1 - self.cth) or
                           (pos[j, 0] < self.cth and
                           pos[i, 0] > 1 - self.cth)):
                            if(abs(pos[i, 1] - pos[j, 1]) < self.r_d and
                               abs(pos[i, 2] - pos[j, 2]) < self.s_d):
                                lungs = ((masks == labels_s[i]) +
                                         (masks == labels_s[j]))
                                lungs_segmented = True
                if not lungs_segmented:
                    dist = np.sum((pos - self.lung_assumed_center_n) ** 2,
                                  axis=1)
                    d_min_idx = np.argmin(dist)
                    lungs = (masks == labels_s[d_min_idx])
                    lungs_segmented = True
        return lungs

    def _remove_outside_body(self, img_patch):

        h, w = img_patch.shape

        M, label = measurements.label(img_patch == 0)
        for l in range(1, label + 1):
            air_region = M == l
            if np.sum(air_region[0, :]):
                img_patch += air_region
            elif np.sum(air_region[:, 0]):
                img_patch += air_region
            elif np.sum(air_region[h - 1, :]):
                img_patch += air_region
            elif np.sum(air_region[:, w - 1]):
                img_patch += air_region

        return img_patch

    def _largest_air_object_center(self, lungs_cs, threshold=2.0):

        h, w, d = lungs_cs.shape
        r = np.zeros((h, 1))
        r[:, 0] = np.arange(h)
        coronal_s = np.sum(r * np.sum(lungs_cs, axis=1), axis=0)
        coronal_s /= (w * h)
        cs_max, cs_max_idx = [np.max(coronal_s), np.argmax(coronal_s)]

        th_idx_low, th_idx_high = [0, d - 1]

        if cs_max < 10:
            threshold = 1.0

        for s in range(cs_max_idx, 0, -1):
            if coronal_s[s] < threshold:
                th_idx_low = s
                break
        for s in range(cs_max_idx, d):
            if coronal_s[s] < threshold:
                th_idx_high = s
                break

        lung_c, lung_sum = [0, 0]
        for s in range(th_idx_low, th_idx_high + 1):
            lung_c += (s * coronal_s[s])
            lung_sum += coronal_s[s]
        lungs_c_s = lung_c / lung_sum

        lungs_bottom = int(th_idx_low - 0.2 * (lungs_c_s - th_idx_low))
        lungs_top = int(th_idx_high + 0.2 * (th_idx_high - lungs_c_s))

        for s in range(d):
            if s < lungs_bottom:
                lungs_cs[:, :, s] = 0.0
            if s > lungs_top:
                lungs_cs[:, :, s] = 0.0

        lungs_c_s /= d
        self.lung_assumed_center_n[2] = lungs_c_s

    def lung_segmentation(self, sc):
        """Method for lungs segmentation."""
        """
            Arguments:
                sc - LiTSscan object containing normalized volume
        """
        h, w, d = [sc.get_height(), sc.get_width(), sc.get_depth()]
        # .....................................................................
        #  1. Detecting air regions around and in body
        # .....................................................................
        air_r = (sc.get_volume() < self.air_th)
        # .....................................................................
        #  2. Detecting body bounds
        # .....................................................................
        b_bounds = np.zeros((sc.get_depth(), 4), dtype='int16')
        b_bounds[:, 1], b_bounds[:, 3] = [w - 1, h - 1]

        bb_v, bb_h = [h - np.sum(air_r, axis=0), w - np.sum(air_r, axis=1)]

        bb_v = bb_v.astype('float32') / np.sum(bb_v, axis=0)
        bb_h = bb_h.astype('float32') / np.sum(bb_h, axis=0)

        xc = np.dot(np.arange(w).astype('float32'), bb_v)
        yc = np.dot(np.arange(h).astype('float32'), bb_h)

        for i in range(d):
            for j in range(int(xc[i]), 0, -1):
                if bb_v[j, i] < self.body_bounds_th[0]:
                    b_bounds[i, 0] = j
                    break
            for j in range(int(xc[i]), w):
                if bb_v[j, i] < self.body_bounds_th[0]:
                    b_bounds[i, 1] = j
                    break
            for j in range(int(yc[i]), 0, -1):
                if bb_h[j, i] < self.body_bounds_th[1]:
                    b_bounds[i, 2] = j
                    break
            for j in range(int(yc[i]), h):
                if bb_h[j, i] < self.body_bounds_th[2]:
                    b_bounds[i, 3] = j
                    break
        # .....................................................................
        #  3. Down-sample air mask for further processing
        # .....................................................................
        air_r_ds = (air_r[::self.ds_f[0], ::self.ds_f[1], ::self.ds_f[2]] == 0)
        # .....................................................................
        #  4. Remove outside body air
        # .....................................................................
        l_cs = np.zeros((h / self.ds_f[0], w / self.ds_f[1], d / self.ds_f[2]),
                        dtype='bool')
        b_bounds_ds = b_bounds
        b_bounds_ds[:, 0:2] = b_bounds_ds[:, 0:2] / self.ds_f[0]
        b_bounds_ds[:, 2:] = b_bounds_ds[:, 2:] / self.ds_f[1]

        for i in range(sc.get_depth()):
            img_patch = air_r_ds[b_bounds_ds[i, 2]: b_bounds_ds[i, 3],
                                 b_bounds_ds[i, 0]: b_bounds_ds[i, 1], i]
            img_patch = self._remove_outside_body(img_patch)
            l_cs[b_bounds_ds[i, 2]:b_bounds_ds[i, 3],
                 b_bounds_ds[i, 0]:b_bounds_ds[i, 1], i] = (img_patch == 0)
        # .....................................................................
        #  5. Determine center of the largest air object along vertical axis
        # .....................................................................
        self._largest_air_object_center(l_cs)
        # .....................................................................
        #  6. Labeling in body air
        # .....................................................................
        masks, label = measurements.label(l_cs)
        labels_list = np.arange(label) + 1
        object_sizes = np.zeros(label)
        count_sgnf = 0
        lv_th_ds = self.lv_th / (self.ds_f[0] * self.ds_f[1] * self.ds_f[2])
        for l in range(1, label + 1):
            object_sizes[l - 1] = np.sum(masks == l)
            if object_sizes[l - 1] > lv_th_ds:
                count_sgnf += 1
        # .....................................................................
        #  7. Extracting lung candidates from labeled data according to the
        #     size and/or position
        # .....................................................................
        lungs = self._extract_lung_candidates(count_sgnf, labels_list, masks,
                                              object_sizes)
        # .....................................................................
        #  8. Up-sample detected mask corresponding to the lungs
        # .....................................................................
        lungs_up = np.zeros((h, w, d), dtype='bool')
        for s in range(d):
            for r in range(h / self.ds_f[0]):
                for c in range(w / self.ds_f[1]):
                    if lungs[r, c, s]:
                        r1, r2 = [r * self.ds_f[0], (r + 1) * self.ds_f[0]]
                        c1, c2 = [c * self.ds_f[1], (c + 1) * self.ds_f[1]]
                        lungs_up[r1:r2, c1:c2, s] = air_r[r1:r2, c1:c2, s]

        # .....................................................................
        #  9. Re-labeling in body air
        # .....................................................................
        masks, label = measurements.label(lungs_up)
        labels_list = np.arange(label) + 1
        object_sizes = np.zeros(label)
        count_sgnf = 0
        for l in range(1, label + 1):
            object_sizes[l - 1] = np.sum(masks == l)
            if object_sizes[l - 1] > self.lv_th:
                count_sgnf += 1
        lungs = self._extract_lung_candidates(count_sgnf, labels_list, masks,
                                              object_sizes)
        # .....................................................................
        #  10. Set meta segmentation
        # .....................................................................
        sc.set_meta_segmentation(lungs * 3)
