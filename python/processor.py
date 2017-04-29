"""Class for volume and segmentation processing."""
import numpy as np
from scan import LiTSscan


class LiTSprocessor(object):
    """LiTSpreprocessor class for volume normalization and flipping.

    Attributes:
        low_threshold (float): lower volume value limit
        high_threshold (float): higher volume value limit
        minimum_value: minimum output value
        maximum_value: maximum output value
        orient: list of orientations
        ord: ordinal number of axes

    Methods:
        preprocess_volume: normalize voxel values and reorient axes
            if necessary
        normalize_volume: normalize voxel values
        reorient_volume: reorient volume if necessary
        reorient_segmentation: reorient segmentation if necessary
        get_axes_orientation: get desired orientation of the axes
        get_axes_order: get desired order of the axes
    """

    def __init__(self, low_threshold=-300.0, high_threshold=700.0,
                 minimum_value=-0.5, maximum_value=0.5,
                 orient_=[1, 1, 1], ord_=[1, 0, 2]):
        """Initialization method for LiTSpreprocess object.

        Arguments:
            low_threshold (float): lower volume value limit
            high_threshold (float): higher volume value limit
            minimum_value: minimum output value
            maximum_value: maximum output value
            orient_: list of orientations
            ord_: ordinal number of axes
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.orient = orient_
        self.ord = ord_

    def preprocess_volume(self, scan):
        """Preprocess volume: clip, normalize values, reorient axes."""
        """
            Arguments:
                scan: LiTSscan object containing volume and information
                    about volume
        """
        volume = scan.get_volume()
        volume = np.clip(volume, self.low_threshold, self.high_threshold)

        volume = ((volume - self.low_threshold) *
                  (self.maximum_value - self.minimum_value) /
                  (self.high_threshold - self.low_threshold) +
                  self.minimum_value)
        if scan.get_axes_orientation()[1] != self.orient[1]:
            volume = volume[::-1, :, :]
        if scan.get_axes_orientation()[2] != self.orient[2]:
            volume = volume[:, :, ::-1]

        tr = []
        for i in range(3):
            for j in range(3):
                if scan.get_axes_order()[i] == self.get_axes_order()[j]:
                    tr.append(j)
        volume = volume.transpose(tr)

        scan.set_volume(volume)

    def normalize_volume(self, scan):
        """Normalize volume: clip and normalize voxel values."""
        """
            Arguments:
                scan: LiTSscan object containing volume and information
                    about volume
        """
        volume = scan.get_volume()
        volume = np.clip(volume, self.low_threshold, self.high_threshold)

        volume = ((volume - self.low_threshold) *
                  (self.maximum_value - self.minimum_value) /
                  (self.high_threshold - self.low_threshold) +
                  self.minimum_value)
        scan.set_volume(volume)

    def reorient_volume(self, input_,
                        cord=None, corient=None,
                        dord=None, dorient=None):
        """Reorient volume: reorient volume axes."""
        """Arguments:
            input_: either LiTSscan object containing volume or
                3d array containing volume
            cord: current order of the volume's axes
            corient: current orientation of the volume's axes
            dord: desired order of the volume's axes
            dorient: desired orientation of the volume's axes
        """
        if isinstance(input_, LiTSscan):
            volume = input_.get_volume()
            cord = input_.get_axes_order()
            dord = self.get_axes_order()
            corient = input_.get_axes_orientation()
            dorient = self.get_axes_orientation()
        else:
            volume = input_

        if corient[1] != dorient[1]:
            volume = volume[::-1, :, :]
        if corient[2] != dorient[2]:
            volume = volume[:, :, ::-1]

        tr = []
        for i in range(3):
            for j in range(3):
                if cord[i] == dord[j]:
                    tr.append(j)
        volume = volume.transpose(tr)

        if isinstance(input_, LiTSscan):
            input_.set_volume(volume)
        else:
            input_ = volume

    def reorient_segmentation(self, input_,
                              cord=None, corient=None,
                              dord=None, dorient=None):
        """Reorient segmentation: reorient segmentation axes."""
        """Arguments:
            input_: either LiTSscan object containing volume or
                3d array containing volume
            cord: current order of the volume's axes
            corient: current orientation of the volume's axes
            dord: desired order of the volume's axes
            dorient: desired orientation of the volume's axes
        """
        if isinstance(input_, LiTSscan):
            segment = input_.get_segmentation()
            cord = input_.get_axes_order()
            dord = self.get_axes_order()
            corient = input_.get_axes_orientation()
            dorient = self.get_axes_orientation()
        else:
            segment = input_

        if corient[1] != dorient[1]:
            segment = segment[::-1, :, :]
        if corient[2] != dorient[2]:
            segment = segment[:, :, ::-1]

        tr = []
        for i in range(3):
            for j in range(3):
                if cord[i] == dord[j]:
                    tr.append(j)
        segment = segment.transpose(tr)

        if isinstance(input_, LiTSscan):
            input_.set_segmentation(segment)
        else:
            input_ = segment

    def get_axes_orientation(self):
        """Get default desired orientation of axes."""
        return self.orient

    def get_axes_order(self):
        """Get default desired order of axes."""
        return self.ord
