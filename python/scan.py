"""Class for scan loading and writing."""
import numpy as np
import nibabel as nib


class LiTSscan(object):
    """LiTSscan class for scan loading and writing.

    Attributes:
        name (str): scan name
        volume (numpy array): 3d array where scan volume is stored
        h (int): volume hight
        w (int): volume width
        d (int): volume depth
        segmentation (numpy array): 3d array where ground truth labels
                                    for liver and tumor are stored
        voxel_size (list): list of floats that represent voxel hight,
                           width and depth (slice distance)
        orientation_info (tuple): tuple of three strings indicating
                                  scaner's coordinate system
    Methods:
        load_volume: loads and stores volume and its hight, width and depth
        load_segmentation: loads and stores ground truth segmentation
        load_info: loads and stores coordinate system orientation info and
                   voxel size
    """

    def __init__(self, name):
        """Initialization method for LiTSscan object.

        Args:
            name (str): scan name
        """
        self.name = name
        self.volume = None
        self.h = None
        self.w = None
        self.d = None
        self.segmentation = None
        self.voxel_size = None
        self.orientation_info = None

    def load_volume(self, s_path):
        """Load and store volume and its hight, width and depth.

        Args:
            s_path (str): path to the volume file
        """
        self.volume = np.array(nib.load(s_path).get_data(), dtype='float32')
        self.h, self.w, self.d = self.volume.shape

    def load_segmentation(self, s_path):
        """Load and store ground truth segmentation.

        Args:
            s_path (str): path to the segmentation file
        """
        self.segmentation = nib.load(s_path).get_data()

    def load_info(self, s_path):
        """Load and store scaner's coordinate system info and voxel size.

        Args:
            s_path (str): path to the volume file
        """
        slice_data = nib.load(s_path)
        self.orientation_info = nib.aff2axcodes(slice_data.affine)
        self.voxel_size = [abs(slice_data.affine[i, i]) for i in range(3)]

    def save_volume(self, s_path):
        """Save volume - to be implemented.

        Args:
            s_path (str): output path to the volume
        """
        raise NotImplementedError()

    def save_segmentation(self, s_path):
        """Save segmentation - to be implemented.

        Args:
            s_path (str): output path to the segmentation
        """
        raise NotImplementedError()
