import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

MIN_LUNG_SIZE_SA = 12000 # mm2
MIN_LUNG_SIZE_LAP = 100  # mm2
AIR_THRESHOLD = -0.45    # volume values normalized between -0.5 and 0.5
MAX_LUNG_LAP_DIST = 100  # mm


def _append_neighbors(objects, mask, i, j, to_verify):

	h, w = objects.shape
	for i_s in range(-1, 2):
		for j_s in range(-1, 2):
			if i_s != 0 or j_s != 0:
				if 0 <= (i + i_s) < h and 0 <= (j + j_s) < w:
					if objects[(i + i_s), (j + j_s)] and not mask[(i + i_s), (j + j_s)]:
						if [(i + i_s), (j + j_s)] not in to_verify:
							to_verify.append([(i + i_s), (j + j_s)])
	return to_verify

def labeling(objects):

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
					to_verify = _append_neighbors(objects, mask, i, j, to_verify)
				if len(to_verify):
					while len(to_verify):
						p = to_verify[0]
						to_verify = to_verify[1:]
						mask[p[0], p[1]] = label
						objects[p[0], p[1]] = 0
						to_verify = _append_neighbors(objects, mask, p[0], p[1], to_verify)

	return mask, label

def region_size(region, voxel_size):

	s = np.sum(region) * voxel_size[0] * voxel_size[1]

	return s

def region_center(region):

	h, w = region.shape
	rs = np.sum(region)

	on_x = np.sum(region, axis=0, dtype=float) / rs
	on_y = np.sum(region, axis=1, dtype=float) / rs

	on_x_c = np.sum(on_x * np.arange(0, w), dtype=float) / w
	on_y_c = np.sum(on_y * np.arange(0, h), dtype=float) / h

	return on_x_c, on_y_c

def region_std(region, xc, yc):

	h, w = region.shape
	rs = np.sum(region)
	on_x = np.sum(region, axis=0, dtype=float) / rs
	on_y = np.sum(region, axis=1, dtype=float) / rs

	std_on_x_c = np.sqrt(np.sum((on_x * (np.arange(0, w) - xc * w)) ** 2, dtype=float)) / w
	std_on_y_c = np.sqrt(np.sum((on_y * (np.arange(0, h) - yc * h)) ** 2, dtype=float)) / h

	return std_on_x_c, std_on_y_c


def lung_detection_sa(right_lung_region_sa, voxel_size):

	candidates = (right_lung_region_sa < AIR_THRESHOLD)
	size_sa_all = region_size(candidates, voxel_size)

	if size_sa_all < MIN_LUNG_SIZE_SA:
		return False, 0

	mask, no_labels = labeling(candidates)

	for l in range(1, no_labels + 1):
		region = (mask==l)
		size_sa = region_size(region, voxel_size)

		if size_sa < MIN_LUNG_SIZE_SA:
			continue

		if size_sa > MIN_LUNG_SIZE_SA:
			return True, region

	return False, 0

def lung_detection_lap(right_lung_rectangle_lap, xc, yc, voxel_size):

	h, w = right_lung_rectangle_lap.shape
	candidates = (right_lung_rectangle_lap < AIR_THRESHOLD)
	mask, no_labels = labeling(candidates)

	label_candidates = []
	for l in range(1, no_labels + 1):

		region = (mask == l)
		s = region_size(region, voxel_size)
		if s < MIN_LUNG_SIZE_LAP:
			continue

		xrc, yrc = region_center(region)

		dist_to_sa = np.sqrt(((xc - xrc * w) * voxel_size[2]) ** 2 + ((yc - yrc * h) * voxel_size[0]) ** 2)

		if dist_to_sa > MAX_LUNG_LAP_DIST:
			continue
		else:
			label_candidates.append(l)

	if len(label_candidates) == 1:
		return mask == label_candidates[0]
	else:
		# should be solved nicer
		std_rc = 1.0
		l_rc = label_candidates[0]
		for l in label_candidates:
			region = (mask == l)
			xrc, yrc = region_center(region)
			if np.abs(yrc - 0.5) > 0.3:
				continue
			std_xrc, std_yrc = region_std(region, xrc, yrc)
			if (std_xrc + std_yrc) / 2 < std_rc:
				std_rc = (std_xrc + std_yrc) / 2
				l_c = l

	return mask==l_c