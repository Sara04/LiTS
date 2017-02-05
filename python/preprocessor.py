"""Class for volume preprocessing."""
import numpy as np
import cv2


class LiTSpreprocessor(object):
    """LiTSpreprocessor class for volume normalization and flipping.

    Attributes:
        low_threshold (float): lower volume value limit
        high_threshold (float): higher volume value limit

    Methods:
        preprocess: volume clipping, normalization and if necessary
                    flipping
    """

    def __init__(self, low_threshold=-300.0, high_threshold=700.0):
        """Initialization method for LiTSpreprocess object.

        Args:
            low_threshold (float): lower volume value limit
            high_threshold (float): higher volume value limit
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def preprocess(self, scan):
        """Preprocess volume.

        Clipping, normalization and posterior -> anterior flip
        if necessary

        Args:
            scan (numpy array): 3d array containing volume
        """
        scan.volume = np.clip(scan.volume,
                              self.low_threshold,
                              self.high_threshold)

        scan.volume = (scan.volume - self.low_threshold) /\
                      (self.high_threshold - self.low_threshold) - 0.5

        if scan.orientation_info[1] == 'P':
            scan.volume = scan.volume[::-1, :, :]
            scan.segmentation = scan.segmentation[::-1, :, :]
