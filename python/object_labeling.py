"""Functions used for object labeling using region growing approach."""
import numpy as np


def _append_neighbors(objects, mask, i, j, to_verify):

    h, w = objects.shape
    for i_s in range(-1, 2):
        for j_s in range(-1, 2):
            if i_s or j_s:
                if 0 <= (i + i_s) < h and 0 <= (j + j_s) < w:
                    if (objects[(i + i_s), (j + j_s)] and
                       not mask[(i + i_s), (j + j_s)]):
                        if [(i + i_s), (j + j_s)] not in to_verify:
                            to_verify.append([(i + i_s), (j + j_s)])
    return to_verify


def region_growing(init_indices, objects):
    """Region growing function.

    Args:
        init_indices (list): list of initial object's indices
        objects (numpy array): binary image

    Returns:
        mask (numpy array): binary image with detected object

    """
    h, w = objects.shape
    mask = np.zeros((h, w))

    to_verify = init_indices

    while len(to_verify):
        p = to_verify[0]
        to_verify = to_verify[1:]
        mask[p[0], p[1]] = 1
        objects[p[0], p[1]] = 0
        to_verify = _append_neighbors(objects, mask,
                                      p[0], p[1],
                                      to_verify)
    return mask


def labeling(objects):
    """Label objects in the binary image.

    Args:
        objects (numpy array): binary image

    Returns:
        mask (numpy array): image with detected objects
        label (int): number of detected objects/labels

    """
    h, w = objects.shape
    mask = np.zeros((h, w))
    label = 0

    while np.sum(objects) != 0:
        to_verify = []
        for i in range(0, h):
            for j in range(0, w):
                if objects[i, j] and not len(to_verify):
                    label += 1
                    mask[i, j] = label
                    objects[i, j] = 0
                    to_verify = _append_neighbors(objects, mask, i, j,
                                                  to_verify)
                if len(to_verify):
                    while len(to_verify):
                        p = to_verify[0]
                        to_verify = to_verify[1:]
                        mask[p[0], p[1]] = label
                        objects[p[0], p[1]] = 0
                        to_verify = _append_neighbors(objects, mask,
                                                      p[0], p[1],
                                                      to_verify)
    return mask, label


def _append_neighbors_3d(objects, mask, i, j, k, to_verify):

    h, w, d = objects.shape
    for i_s in range(-1, 2):
        for j_s in range(-1, 2):
            for k_s in range(-1, 2):
                if i_s or j_s or k_s:
                    if (0 <= (i + i_s) < h and
                       0 <= (j + j_s) < w and
                       0 <= (k + k_s) < d):

                        if (objects[(i + i_s),
                                    (j + j_s),
                                    (k + k_s)] and
                           not mask[(i + i_s),
                                    (j + j_s),
                                    (k + k_s)]):
                            if ([(i + i_s),
                                 (j + j_s),
                                 (k + k_s)]
                               not in to_verify):
                                to_verify.\
                                    append([(i + i_s),
                                            (j + j_s),
                                            (k + k_s)])
    return to_verify


def region_growing_3d(init_slice, objects):
    """Region growing method for 3d binary image.

    Args:
        init_indices (list): list of initial indices
        objects (numpy array): 3d binary image

    Returns:
        mask (numpy array): 3d image with detected object

    """
    h, w, d = objects.shape
    mask = np.zeros((h, w, d))

    to_verify = init_slice

    while len(to_verify):
        p = to_verify[0]
        to_verify = to_verify[1:]
        mask[p[0], p[1], p[2]] = 1
        objects[p[0], p[1], p[2]] = 0
        to_verify = _append_neighbors_3d(objects, mask,
                                         p[0], p[1], p[2],
                                         to_verify)
    return mask
