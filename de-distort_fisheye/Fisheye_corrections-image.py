"""
Credits : https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
"""

import numpy as np
import cv2
from tqdm.notebook import tqdm
import pickle


# You should replace these 3 lines with the output in calibration step
DIM = (1088, 1080)
K = np.array([[773.6719811071623, 0.0, 532.3446174857597], [0.0, 774.3187867828567, 565.9954169588382], [0.0, 0.0, 1.0]])
D = np.array([[0.007679273278292513], [-0.1766120836825416], [0.6417799798538761], [-0.7566634368371957]])

def undistort(img, balance=0.0, dim2=None, dim3=None):

	dim1 = img.shape[:2][::-1]
	#dim1 is the dimension of input image to un-distort
	assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
	if not dim2:
		dim2 = dim1
	if not dim3:
		dim3 = dim1

	scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
	scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
	# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
	new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim1, np.eye(3), balance=balance)
	map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2)
	undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

	return undistorted_img


ratio_image = 1.5
image_file_path = '../input/'
image_file_name = 'PI world v1 ps1_181.jpg'

image_file = image_file_path + image_file_name

image = cv2.imread(image_file)

Image_name = "Image"
cv2.namedWindow(Image_name)

# width, height, rgb = np.shape(image)

undistorted_image = undistort(image, balance=0.0)

with open(f'../output/{image_file_name[:-4]}_undistorted_images.pkl', 'wb') as handle:
	pickle.dump(undistorted_image, handle)


cv2.imshow(Image_name, undistorted_image)
while True:
	key = cv2.waitKey(1) & 0xFF




