import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Preprocessor(object):

	def __init__(self, low_threshold=-300, high_threshold=700, normalization='min_max'):
		
		self.low_threshold = low_threshold
		self.high_threshold = high_threshold
		self.normalization = normalization

	def preprocess(self, volume):
		volume = np.clip(volume, self.low_threshold, self.high_threshold)

		volume_n = (volume - self.low_threshold) / (self.high_threshold - self.low_threshold)

		volume_n -= 0.5

		return volume_n
