import os
import numpy as np
import nibabel as nib
from database import LiTSdb
from preprocessor import Preprocessor
from classifier import CNNClassifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from region_detectors import *
from organ_detectors import *
import time

MIN_BODY_SLICE_RATIO = 0.2

def main():	

	path_to_database = '/home/sara/science/databases/LiTS/Training Batch 1/Training Batch 1'

	db = LiTSdb(path_to_database)
	preprocess = Preprocessor()

	subjects = db.get_subjects_name()
	data = {}

	for s in subjects[0:]:

		print("s:", s)

		if s not in data:
			data[s] = {}

		# 1. Loading CT volume, segmentation and voxel size
		#________________________________________________________________________________________________________________________#
		volume = db.load_volume(s)
		segmentation = db.load_segmentation(s)
		volume = preprocess.preprocess(volume)
		voxel_size = db.load_voxel_size(s)
		h, w, n = volume.shape
		#________________________________________________________________________________________________________________________#
		# 2. Detecting liver tissue
		#________________________________________________________________________________________________________________________#
		lung_detected_sa = False
		for i in range(1, n):
			# 2.1. Find body bbox in short axis view
			#____________________________________________________________________________________________________________________#
			x_start, x_end, y_start, y_end = body_bbox_sa(volume[:, :, n - i])
			ratio = float(y_end - y_start) * float(x_end - x_start) / (h * w)
			if ratio < MIN_BODY_SLICE_RATIO:
				continue
			#____________________________________________________________________________________________________________________#
			# 2.2 Detect liver (liver initialization)
			#____________________________________________________________________________________________________________________#
			if not lung_detected_sa:
				# 2.2.1 Set right lung approximate bbox and try to detect large lung region
				#________________________________________________________________________________________________________________#
				x_start_rl_sa, x_end_rl_sa, y_start_rl_sa, y_end_rl_sa = [x_start, x_end, y_start, (y_start + y_end) / 2]
				right_lung_bbox_sa = extract_bbox(volume[:, :, n - i], x_start_rl_sa, x_end_rl_sa, y_start_rl_sa, y_end_rl_sa)
				lung_detected_sa, lung_region_sa = lung_detection_sa(right_lung_bbox_sa, voxel_size)
				#________________________________________________________________________________________________________________#
			else:
				# 2.2.2 Set right lung approximate bbox and try to detect large lung region
				#________________________________________________________________________________________________________________#
				h_rl_sa, w_rl_sa = lung_region_sa.shape
				xc_rl_sa, yc_rl_sa = region_center(lung_region_sa)
				right_lung_bbox_lap = extract_bbox(volume[y_start_rl_sa + yc_rl_sa * h_rl_sa, :, :], 0, -1, x_start_rl_sa, x_end_rl_sa)
				lung_region_lap = lung_detection_lap(right_lung_bbox_lap, n - i, yc_rl_sa * w_rl_sa, voxel_size)
				plt.figure(1)
				plt.imshow(lung_region_lap)
				plt.show()
				break
				#________________________________________________________________________________________________________________#
		#________________________________________________________________________________________________________________________#

		data[s]['volume'] = volume
		data[s]['segmentation'] = segmentation
		data[s]['voxel_size'] = voxel_size


	s_train = [subjects[i] for i in range(0, 20)]
	s_valid = [subjects[i] for i in range(20, 26)]
	s_test = [subjects[i] for i in range(26, 27)]

	cnn = CNNClassifier()

	cnn.train(data, s_train, s_valid)
	cnn.test(data, s_test)

if __name__ == '__main__':
	main()