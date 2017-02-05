"""Functions used for calculation statistical properties of an object."""
import numpy as np


def region_center(region):
    """Compute center of binary region.

    Args:
        region (numpy array): binary image containing region

    Returns:
        centers along y and x axis on scale from 0 to 1
    """
    h, w = region.shape
    rs = np.sum(region)

    on_x = np.sum(region, axis=0, dtype=float) / rs
    on_y = np.sum(region, axis=1, dtype=float) / rs

    on_x_c = np.sum(on_x * np.arange(0, w), dtype=float) / w
    on_y_c = np.sum(on_y * np.arange(0, h), dtype=float) / h

    return on_y_c, on_x_c


def volume_center_3d(volume):
    """Compute center of binary volume.

    Args:
        volume (numpy array): 3d binary image containing volume

    Returns:
        centers along y, x and z axis on scale from 0 to 1
    """
    h, w, d = volume.shape
    v = np.sum(volume)

    on_x = np.sum(np.sum(volume, axis=2), axis=0, dtype=float) / v
    on_y = np.sum(np.sum(volume, axis=2), axis=1, dtype=float) / v
    on_z = np.sum(np.sum(volume, axis=1), axis=0, dtype=float) / v

    on_x_c = np.sum(on_x * np.arange(0, w), dtype=float) / w
    on_y_c = np.sum(on_y * np.arange(0, h), dtype=float) / h
    on_z_c = np.sum(on_z * np.arange(0, d), dtype=float) / d

    return on_y_c, on_x_c, on_z_c
