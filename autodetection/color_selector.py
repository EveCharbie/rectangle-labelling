"""
Credits : https://blog.electroica.com/hsv-trackbar-opencv-python/
"""

import cv2
import numpy as np
import pickle

def callback(x):
    global H_low, H_high, S_low, S_high, V_low, V_high
    H_low = cv2.getTrackbarPos('low H', 'controls')
    H_high = cv2.getTrackbarPos('high H', 'controls')
    S_low = cv2.getTrackbarPos('low S', 'controls')
    S_high = cv2.getTrackbarPos('high S', 'controls')
    V_low = cv2.getTrackbarPos('low V', 'controls')
    V_high = cv2.getTrackbarPos('high V', 'controls')


# Image_name = "../input/Back.png"
Image_name = '../input/PI world v1 ps1_181.jpg'
img = cv2.imread(Image_name)

# movie_path = "../output/"
# movie_name = "PI world v1 ps1.mp4"
# movie_file = movie_path + movie_name[:-4] + "_undistorted_images.pkl"
# file = open(movie_file, "rb")
# frames = pickle.load(file)
# img = frames[0]


Image_name = '../output/PI world v1 ps1_181_undistorted_images.pkl'
file = open(Image_name, "rb")
img = pickle.load(file)

cv2.namedWindow('controls', 2)
cv2.resizeWindow("controls", 550, 10)

H_low = 0
H_high = 179
S_low = 0
S_high = 255
V_low = 0
V_high = 255

# create trackbars for high,low H,S,V
cv2.createTrackbar('low H', 'controls', 0, 179, callback)
cv2.createTrackbar('high H', 'controls', 179, 179, callback)

cv2.createTrackbar('low S', 'controls', 0, 255, callback)
cv2.createTrackbar('high S', 'controls', 255, 255, callback)

cv2.createTrackbar('low V', 'controls', 0, 255, callback)
cv2.createTrackbar('high V', 'controls', 255, 255, callback)

while (1):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_low = np.array([H_low, S_low, V_low], np.uint8)
    hsv_high = np.array([H_high, S_high, V_high], np.uint8)

    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    res = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()