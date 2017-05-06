"""Class for volume and segmentation processing."""
import numpy as np
from scan import LiTSscan
import tensorflow as tf
import sys


class LiTSprocessor(object):
    """LiTSpreprocessor class for volume normalization and flipping.

    Attributes:
        low_th (float): lower volume value limit
        high_th (float): higher volume value limit
        min_value: minimum output value
        max_value: maximum output value
        orient: list of orientations
        ord: ordinal number of axes
        approach: cpu/gpu processing/normalization/reorient approach

    Methods:
        preprocess_volume: normalize voxel values and reorient axes
            if necessary
        normalize_volume: normalize voxel values
        reorient_volume: reorient volume if necessary
        reorient_segmentation: reorient segmentation if necessary
        get_axes_orientation: get desired orientation of the axes
        get_axes_order: get desired order of the axes
    """

    def __init__(self, low_th=-300.0, high_th=700.0,
                 min_value=-0.5, max_value=0.5,
                 orient_=[1, 1, 1], ord_=[1, 0, 2], approach_='gpu'):
        """Initialization method for LiTSpreprocess object.

        Arguments:
            low_th (float): lower volume value limit
            high_th (float): higher volume value limit
            min_value: min output value
            max_value: max output value
            orient_: list of orientations
            ord_: ordinal number of axes
            approach_: cpu/gpu processing/normalization/reorient approach
        """
        self.low_th = low_th
        self.high_th = high_th
        self.min_value = min_value
        self.max_value = max_value
        self.orient = orient_
        self.ord = ord_
        self.approach = approach_

        if self.approach == 'gpu':
            self.session = tf.Session()
            self.volume_tf = tf.placeholder('float32', [None, None, None])
            self.segment_tf = tf.placeholder('uint8', [None, None, None])

    def _normalize_voxel_values_gpu(self, volume):

        h, w, d = volume.shape
        v_tf = tf.placeholder('float32', [h, w, d])
        v_n_gpu = ((tf.clip_by_value(v_tf, self.low_th, self.high_th) -
                    self.low_th) *
                   (self.max_value - self.min_value) /
                   (self.high_th - self.low_th) + self.min_value)

        return self.session.run(v_n_gpu, feed_dict={v_tf: volume})

    def _preprocess_volume_gpu(self, volume, axes_orient, axes_tranpose):

        h, w, d = volume.shape
        v_p_gpu = ((tf.clip_by_value(self.volume_tf,
                                     self.low_th, self.high_th) -
                    self.low_th) *
                   (self.max_value - self.min_value) /
                   (self.high_th - self.low_th) + self.min_value)

        dims = []
        if axes_orient[1] != self.orient[1]:
            dims.append(0)
        if axes_orient[2] != self.orient[2]:
            dims.append(2)

        v_p_gpu = tf.reverse(v_p_gpu, dims)
        v_p_gpu = tf.transpose(v_p_gpu, axes_tranpose)

        return self.session.run(v_p_gpu, feed_dict={self.volume_tf: volume})

    def _reorient_volume_gpu(self, volume, axes_orient, axes_tranpose):

        dims = []
        if axes_orient[1] != self.orient[1]:
            dims.append(0)
        if axes_orient[2] != self.orient[2]:
            dims.append(2)

        v_r_gpu = tf.reverse(self.volume_tf, dims)
        v_r_gpu = tf.transpose(v_r_gpu, axes_tranpose)

        return self.session.run(v_r_gpu, feed_dict={self.volume_tf: volume})

    def _reorient_segment_gpu(self, segment, axes_orient, axes_tranpose):

        dims = []
        if axes_orient[1] != self.orient[1]:
            dims.append(0)
        if axes_orient[2] != self.orient[2]:
            dims.append(2)

        s_r_gpu = tf.reverse(self.segment_tf, dims)
        s_r_gpu = tf.transpose(s_r_gpu, axes_tranpose)

        return self.session.run(s_r_gpu, feed_dict={self.segment_tf: segment})

    def preprocess_volume(self, scan):
        """Preprocess volume: clip, normalize values, reorient axes."""
        """
            Arguments:
                scan: LiTSscan object containing volume and information
                    about volume
        """
        volume = scan.get_volume()
        tr = []
        for i in range(3):
            for j in range(3):
                if scan.get_axes_order()[i] == self.get_axes_order()[j]:
                    tr.append(j)

        if self.approach == 'cpu':
            volume = np.clip(volume, self.low_th, self.high_th)

            volume = ((volume - self.low_th) *
                      (self.max_value - self.min_value) /
                      (self.high_th - self.low_th) +
                      self.min_value)
            volume = volume.transpose(tr)

            if scan.get_axes_orientation()[1] != self.orient[1]:
                volume = volume[::-1, :, :]
            if scan.get_axes_orientation()[2] != self.orient[2]:
                volume = volume[:, :, ::-1]

        elif self.approach == 'gpu':
            volume = self._preprocess_volume_gpu(volume,
                                                 scan.get_axes_orientation(),
                                                 tr)
        else:
            print("Invalid preprocess approach")
            sys.exit(2)

        scan.set_volume(volume)

    def normalize_volume(self, scan):
        """Normalize volume: clip and normalize voxel values."""
        """
            Arguments:
                scan: LiTSscan object containing volume and information
                    about volume
        """
        volume = scan.get_volume()
        if self.approach == 'cpu':
            volume = np.clip(volume, self.low_th, self.high_th)

            volume = ((volume - self.low_th) *
                      (self.max_value - self.min_value) /
                      (self.high_th - self.low_th) +
                      self.min_value)
        elif self.approach == 'gpu':
            volume = self._normalize_voxel_values_gpu(volume)
        else:
            print("Invalid normalization approach.")
            sys.exit(2)

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

        tr = []
        for i in range(3):
            for j in range(3):
                if cord[i] == dord[j]:
                    tr.append(j)

        if self.approach == 'cpu':
            if corient[1] != dorient[1]:
                volume = volume[::-1, :, :]
            if corient[2] != dorient[2]:
                volume = volume[:, :, ::-1]
            volume = volume.transpose(tr)
        elif self.approach == 'gpu':
            volume = self._reorient_volume_gpu(volume, corient, tr)
        else:
            print("Invalid volume reorientation approach.")
            sys.exit(2)

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

        tr = []
        for i in range(3):
            for j in range(3):
                if cord[i] == dord[j]:
                    tr.append(j)

        if self.approach == 'cpu':
            if corient[1] != dorient[1]:
                segment = segment[::-1, :, :]
            if corient[2] != dorient[2]:
                segment = segment[:, :, ::-1]
            segment = segment.transpose(tr)
        elif self.approach == 'gpu':
            segment = self._reorient_segment_gpu(segment, corient, tr)

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
