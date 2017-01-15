import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

AIR_THRESHOLD = -0.45    # volume values normalized between -0.5 and 0.5

def draw_square(volume, square, color=[0.5, 0.0, 0.0]):
	
	h, w = volume.shape
	img_display = np.zeros((h, w, 3))
	img_display[:, :, 0] = volume
	img_display[:, :, 1] = volume
	img_display[:, :, 2] = volume

	img_display[square[2], square[0]:square[1]] = color
	img_display[square[3], square[0]:square[1]] = color
	img_display[square[2]:square[3], square[0]] = color
	img_display[square[2]:square[3], square[1]] = color

	plt.figure(1)
	plt.imshow(img_display)
	plt.show()


def body_bbox_sa(volume_slice, threshold=0.2, l=10):

	h, w = volume_slice.shape
	mask = (volume_slice > AIR_THRESHOLD)
	on_x_sum = np.mean(mask, axis=0)
	on_y_sum = np.mean(mask, axis=1)

	x_start = 0
	x_end = 0
	for i in range(0, w - l):
		if np.mean(on_x_sum[i:(i+l)]) > threshold:
			x_start = i
			break
	for i in range(w-l, 0, -1):
		if np.mean(on_x_sum[i:(i+l)]) > threshold:
			x_end = i
			break

	y_start = 0
	y_end = 0
	for i in range(0, h - l):
		if np.mean(on_y_sum[i:(i+l)]) > threshold:
			y_start = i
			break
	for i in range(h-l, 0, -1):
		if np.mean(on_y_sum[i:(i+l)]) > threshold:
			y_end = i
			break

	return x_start, x_end, y_start, y_end

def extract_bbox(volume, x_start, x_end, y_start, y_end):

	return volume[y_start:y_end, x_start:x_end]
