import os
import nibabel as nib

class LiTSdb(object):

	def __init__(self, db_name):

		self.db_name = db_name

	def get_subjects_name(self):

		files = os.listdir(self.db_name)

		subjects = []
		for fi in files:
			if fi.startswith('.'):
				continue
			s = str.split(fi, '-')[1][:-4]
			if s not in subjects:
				subjects.append(s)

		return subjects

	def get_volume_paths(self, subjects):
		volume_paths = {}
		for s in subjects:
			volume_path = os.path.join(self.db_name, '-'.join(['volume', s]) + '.nii')
			volume_paths[s] = volume_path

		return volume_paths

	def get_segmentation_paths(self, subjects):
		segment_paths = {}
		for s in subjects:
			segment_path = os.path.join(self.db_name, '-'.join(['segmentation', s]) + '.nii')
			segment_paths[s] = segment_path

		return segment_paths

	def load_volume(self, s):
		volume_path = os.path.join(self.db_name, '-'.join(['volume', s]) + '.nii')
		volume = nib.load(volume_path).get_data()
		return volume

	def load_segmentation(self, s):
		segmentation_path = os.path.join(self.db_name, '-'.join(['segmentation', s]) + '.nii')
		segmentation = nib.load(segmentation_path).get_data()
		return segmentation

	def load_voxel_size(self, s):

		volume_path = os.path.join(self.db_name, '-'.join(['volume', s]) + '.nii')
		all_info = nib.load(volume_path).get_sform()

		return [abs(all_info[0, 0]), abs(all_info[1, 1]), abs(all_info[2, 2])]


