"""Class for scan loading and writing."""
import numpy as np
import nibabel as nib


class LiTSscan(object):
    """LiTSscan class for scan loading and writing.

    Attributes:
        volume_path: path to the volume data
        segmentation_path: path to the ground truth segmentation data
        meta_segmentation_path: path to the meta segmentation data

        volume: volume data
        segmentation: segmentation data
        meta_segmentation: meta segmentation data

        h: volume height (front-back body direction)
        w: volume width (left-right body direction)
        d: volume depth (bottom-top body direction)

        h_voxel: voxel height (front-back body direction)
        w_voxel: voxel width (left-right body direction)
        d_voxel: voxel depth (bottom-top body direction)

        header: nii image header
        affine_m: affine matrix

        axes_order: order of the volume's axes
        axes_orientation: orientation of the volume's axes

    Methods:
        load_volume: loads and stores volume and its hight, width and depth
        load_segmentation: loads and stores ground truth segmentation
        load_info: loads and stores coordinate system orientation info and
                   voxel size

        get_volume: get pointer to the volume data
        get_segmentation: get pointer to the segmentation data
        get_meta_segmentaion: get pointer to the meta segmentation data

        get_height: get height of the volume/segmentation
            (front-back body direction)
        get_width: get width of the volume/segmentation
            (left-right body direction)
        get_depth: get depth of the volume/segmentation
            (bottom-top body direction)

        get_voxel_height: get height of the voxels
        get_voxel_width: get width of the voxels
        get_voxel_depth: get depth of the voxels (slice distance)

        get_axes_order: get the order of the axes
        get_axes_orientation: get the orientations of the axes
    """

    def __init__(self, volume_path, segmentation_path=None,
                 meta_segmentation_path=None):
        """Initialization method for LiTSscan object.

        Arguments:
            volume_path: path to the volume data
            segmentation_path: path to the ground truth segmentation data
            meta_segmentation_path: path to the meta segmentation data
        """
        self.volume_path = volume_path
        self.segmentation_path = segmentation_path
        self.meta_segmentation_path = meta_segmentation_path

        self.volume = None
        self.segmentation = None
        self.meta_segmentation = None

        self.h = None
        self.w = None
        self.d = None

        self.voxel_h = None
        self.voxel_w = None
        self.voxel_d = None

        self.axes_order = None
        self.axes_orientation = None

        self.header = None
        self.affine_m = None

    def load_volume(self):
        """Load and store volume and its hight, width and depth."""
        self.volume = np.array(nib.load(self.volume_path).get_data(),
                               dtype='float32').transpose(1, 0, 2)

    def load_segmentation(self):
        """Load and store ground truth segmentation."""
        self.segmentation =\
            np.array(nib.load(self.segmentation_path).get_data(),
                     dtype='uint8').transpose(1, 0, 2)

    def load_info(self):
        """Load scan's axes order and orientation, volume and voxel sizes."""
        slice_data = nib.load(self.volume_path)
        self.header = slice_data.header
        volume_d = self.header.get_data_shape()
        voxel_s = []
        self.affine_m = slice_data.affine
        inv = [-1, -1, 1]

        self.axes_order = []
        self.axes_orientation = []
        for i in range(3):
            for j in range(3):
                if self.affine_m[i, j] != 0:
                    self.axes_order.append(j)
                    voxel_s.append((self.affine_m[i, j]))
                    self.axes_orientation.append(2 * (self.affine_m[i, j] *
                                                      inv[j] > 0) - 1)

        self.w = volume_d[self.axes_order[0]]
        self.h = volume_d[self.axes_order[1]]
        self.d = volume_d[self.axes_order[2]]

        t = self.axes_order[1]
        self.axes_order[1] = self.axes_order[0]
        self.axes_order[0] = t
        self.voxel_w, self.voxel_h, self.voxel_d = voxel_s

    def set_volume(self, volume):
        """Set volume."""
        """
            Arguments:
                volume: new volume data
        """
        self.volume = volume

    def set_segmentation(self, segmentation):
        """Set segmentation."""
        """
            Arguments:
                segmentation: new segmentation data
        """
        self.segmentation = segmentation

    def set_meta_segmentation(self, meta_segmentation):
        """Set meta segmentation."""
        """
            Arguments:
                meta_segmentation: new meta_segmentation data
        """
        self.meta_segmentation = meta_segmentation

    def get_volume(self):
        """Get volume data."""
        return self.volume

    def get_segmentation(self):
        """Get segmentation data."""
        return self.segmentation

    def get_meta_segmentation(self):
        """Get meta segmentation data."""
        return self.meta_segmentation

    def get_width(self):
        """Get scan width."""
        return self.w

    def get_height(self):
        """Get scan height."""
        return self.h

    def get_depth(self):
        """Get scan depth."""
        return self.d

    def get_voxel_width(self):
        """Get voxel width."""
        return self.voxel_w

    def get_voxel_height(self):
        """Get voxel height."""
        return self.voxel_h

    def get_voxel_depth(self):
        """Get voxel depth."""
        return self.voxel_d

    def get_axes_order(self):
        """Get axes order."""
        return self.axes_order

    def get_axes_orientation(self):
        """Get axes orientation."""
        return self.axes_orientation

    def save_meta_segmentation(self, meta_segmentation_path):
        """Save meta segmentation."""
        """
            Arguments:
                meta_segmentation_path: meta segmentation path
        """
        meta_segment_nib = nib.Nifti1Image(self.get_meta_segmentation(),
                                           self.affine_m, self.header)
        meta_segment_nib.set_data_dtype(np.uint8)
        nib.save(meta_segment_nib, meta_segmentation_path)
