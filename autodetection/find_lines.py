
import cv2
import numpy as np
from tqdm.notebook import tqdm
import pickle



def circle_positioning(event, x, y, flags, param):
    global points_labels, current_click, frame_counter

    if event == cv2.EVENT_LBUTTONDOWN:
        points_labels[label_keys[current_click]][:, frame_counter] = np.array([x, y])
        draw_points_and_lines()
    return



############################### code beginning #######################################################################
# global small_image, image, image_bidon

# circle_radius = 5
# rectangle_color = (1, 1, 1)
# circle_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
#                  (100, 0, 0), (0, 100, 0), (0, 0, 100), (100, 100, 0)]

Image_name = "Video"
Trackbar_name = "Frames"
ratio_image = 1.5

# movie_file_path = '../input/'
# movie_file_name = 'PI world v1 ps1.mp4'
# movie_file = movie_file_path + movie_file_name
#
# file = open(f"../output/{movie_file_name[:-4]}_undistorted_images.pkl", "rb")
# frames = pickle.load(file)

# frames_clone = frames.copy()

def nothing(x):
    return

# cv2.namedWindow(Image_name)
# cv2.createTrackbar('Frames', Image_name, 0, num_frames, nothing)




###################   Methode 1   ##############################

# image = cv2.imread(Image_name, cv2.IMREAD_COLOR)
# image_gray = cv2.imread(Image_name, cv2.IMREAD_GRAYSCALE)
# edged = cv2.Canny(image_gray, 0, 255)
# ret, thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL
# for cnt in contours:
#     moment = cv2.moments(cnt)
#     area = cv2.contourArea(cnt)
#     # if area > 400:
#     approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
#     cv2.drawContours(image_gray, [approx], 0, (0, 0, 255), 5)
#     cv2.drawContours(image_gray, [cnt], -1, (0, 255, 0), 2)
#     cv2.imshow('polygons_detected', image_gray)
# cv2.waitKey(0) & 0xFF


###################   Methode 2 : fonctione a peu pres   ##############################
file = open(f"../output/PI world v1 ps1_181_undistorted_images.pkl", "rb")
frame = pickle.load(file)
image = frame
# image = frames[0]
# Image_name = '../input/PI world v1 ps1_181.jpg'
# image = cv2.imread(Image_name)

# lower = np.array([107, 24, 103], dtype="uint8")
# upper = np.array([169, 97, 146], dtype="uint8")
# lower = np.array([86, 0, 89], dtype="uint8")
# upper = np.array([179, 116, 139], dtype="uint8")
lower = np.array([0, 0, 115], dtype="uint8")
upper = np.array([104, 32, 171], dtype="uint8")

# cv2.imshow("Base image", image)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)
image_mask = cv2.bitwise_and(hsv, hsv, mask=mask)
# cv2.imshow("Result colors", image_mask)
# cv2.waitKey(0) & 0xFF


# image = cv2.imread(Image_name)
image_mask_rgb = cv2.cvtColor(image_mask, cv2.COLOR_HSV2RGB)
image_gray = cv2.cvtColor(image_mask_rgb, cv2.COLOR_RGB2GRAY)
# image_gray = cv2.imread(image_mask, cv2.IMREAD_GRAYSCALE)

# edges = cv2.Canny(image_gray, 50, 150) # apertureSize=3
# minLineLength = 500 # 5000
# maxLineGap = 5
# lines = cv2.HoughLinesP(edges, 1, 2*np.pi / 180, 50, minLineLength, maxLineGap)
# # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength, maxLineGap)
# for x in range(0, len(lines)):
#     for x1, y1, x2, y2 in lines[x]:
#         cv2.line(image, (x1, y1), (x2, y2), (0, 128, 0), 2)
# # cv2.imshow("Result lines", image)
# # cv2.waitKey(0) & 0xFF



###################   Methode 3 : polygone detection  ##############################
# image = frames[0]
#
# lower = np.array([68, 0, 103], dtype="uint8")
# upper = np.array([158, 103, 202], dtype="uint8")
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv, lower, upper)
# image_mask = cv2.bitwise_and(hsv, hsv, mask=mask)
# # cv2.imshow("Result colors", image_mask)
# # cv2.waitKey(0) & 0xFF
#
#
# image_mask_rgb = cv2.cvtColor(image_mask, cv2.COLOR_HSV2RGB)
# image_gray = cv2.cvtColor(image_mask_rgb, cv2.COLOR_RGB2GRAY)

# edged = cv2.Canny(image_gray, 50, 150)
# ret, thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL
# for cnt in contours:
#     # moment = cv2.moments(cnt)
#     # area = cv2.contourArea(cnt)
#
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.boxPoints(rect)
#     area = cv2.contourArea(box)
#     ratio = area / 1000000
#     if ratio < 0.015: # Any contour large enough is a candidate
#         continue
#     cv2.polylines(image, [box.astype(int)], True, (0, 128, 0), 5)
#     # if area > 1:
#     # approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
#     # cv2.drawContours(image, [approx], 0, (0, 128, 0), 5)
#     # cv2.drawContours(image_gray, [cnt], -1, (0, 255, 0), 2)
#     cv2.imshow('polygons_detected', image)
# cv2.waitKey(0) & 0xFF


################################################################################

#
# undistorted_images = [np.zeros((DIM[0], DIM[1], 3), dtype=np.int8) for i in range(num_frames)]
# for i in range(num_frames):
#     # small_images[i] = cv2.resize(frames_clone[i], (int(round(width / ratio_image)), int(round(height / ratio_image))))
#     undistorted_images[i] = undistort(frames[i], balance=0.0)  # small_images[i]
#
# with open(f'../output/{movie_file_name[:-4]}_undistorted_images.pkl', 'wb') as handle:
#     pickle.dump(undistorted_images, handle)

#
# playVideo = True
# frame_counter = 0
# image_clone = frames[0].copy()
# width, height, rgb = np.shape(image_clone)
# while playVideo:
#
#     key = cv2.waitKey(1) & 0xFF
#
#     frame_counter = cv2.getTrackbarPos(Trackbar_name, Image_name)
#     cv2.imshow(Image_name, undistorted_images[frame_counter])
#
#     if key == ord(','):  # if `<` then go back
#         frame_counter -= 1
#         cv2.setTrackbarPos(Trackbar_name, Image_name, frame_counter)
#         cv2.imshow(Image_name, undistorted_images[frame_counter])
#
#     elif key == ord('.'):  # if `>` then advance
#         print("frame_counter fleche +: ", frame_counter, ' -> ', frame_counter+1)
#         frame_counter += 1
#         cv2.setTrackbarPos(Trackbar_name, Image_name, frame_counter)
#         cv2.imshow(Image_name, undistorted_images[frame_counter])
#
#
#     elif key == ord('x'):  # if `x` then quit
#         playVideo = False
#
# cv2.destroyAllWindows()


# ###################   Methode 4 : lignes en combinaison avec méthode 2  ##############################
#
### pourrait être fait avec plusieurs points et un threashold pour exclure des points en RMSD genre
# import matplotlib.pyplot as plt
#
# # Load the image in, convert to black and white
# image = image_gray  # image_mask
# plt.subplot(2, 2, 1)
# plt.imshow(image, cmap="gray")
# plt.title("Back and White Source Image")
#
# image_to_compute = image  # only do this to make experimenting with commenting out parts later
#
# # img_sobel_x = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3)
# # img_sobel_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3)
# # img_sobel = np.abs(img_sobel_x) + np.abs(img_sobel_y)
# # _, image_to_compute = cv2.threshold(img_sobel, .001, 1, cv2.THRESH_BINARY)
# # plt.subplot(2, 2, 2)
# # plt.imshow(image_to_compute, cmap="gray")
# # plt.title("Thresholded Sobel of Input")
# # # Morphology on image
# # # again, this part may not be needed, but will help with noisy images
# # kernel = np.ones((9, 9), np.uint8)
# # closing = cv2.morphologyEx(image_to_compute, cv2.MORPH_CLOSE, kernel)
# # image_to_compute = (closing * 255).astype(np.uint8)
# # plt.subplot(2, 2, 3)
# # plt.imshow(image_to_compute, cmap="gray")
# # plt.title("Morphlogy on Image")
#
#
# # Get the lines as 1 pixel wide using skeletonization
# # Making all lines skinny makes the compute go faster
# def skeletonize(img):
#     '''
#     https://medium.com/analytics-vidhya/skeletonization-in-python-using-opencv-b7fa16867331
#     Get all the lines as 1 pixel wide
#     '''
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     skel = np.zeros(img.shape, np.uint8)
#     # Repeat steps 2-4
#     while True:
#         # Step 2: Open the image
#         opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
#         # Step 3: Substract open from the original image
#         temp = cv2.subtract(img, opening)
#         # Step 4: Erode the original image and refine the skeleton
#         eroded = cv2.erode(img, element)
#         skel = cv2.bitwise_or(skel, temp)
#         img = eroded.copy()
#         # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
#         if cv2.countNonZero(img) == 0:
#             break
#     return skel
#
#
# skel = skeletonize(image_to_compute)
# plt.subplot(2, 2, 4)
# plt.imshow(skel, cmap="gray")
# plt.title("Skeletonized Image")
#
# plt.show()
#
# # Now get the indicies where the pixels have a line in them
# y, x = np.where(skel > 0)
# locs = np.array((x, y), dtype=np.float32).T
# print("Have {} pixels to find lines through".format(locs.shape))
#
#
# # Use a RANSAC algorithm to compute the lines
# # This will randomly sample points, compute the line through them
# #    then find how many pixels are within the max_devaition parameter
# # The more points within the max_deviation parameter the better the line fits
# def RANSAC_LINEST(points, max_deviation=2., n_points=3):
#     '''
#     Approximate a line from the data
#     Returns the (slope,y_intercept) of the line that has the most points within max_deviation of the line
#     '''
#
#     mask_probabilities = ~points[:, 0].mask / points[:, 0].count()
#     best_coeffs = [0, 0]
#     best_count = 0
#     best_idxs = []
#     for it in range(100):
#         # choose random points, and compute the line of best fit for them
#         idxs = np.random.choice(points.shape[0], size=n_points, p=mask_probabilities, replace=False)
#         pts = points[idxs, :]
#         poly = np.poly1d(np.polyfit(pts[:, 0], pts[:, 1], deg=1))
#
#         # compute the resulting points and find matches
#         computed_ys = poly(points[:, 0])
#         deltas = np.abs(points[:, 1] - computed_ys)
#         good_idxs = np.where(deltas < max_deviation)[0]
#         if len(good_idxs) > best_count:
#             best_count = len(good_idxs)
#             best_coeffs = poly.coefficients
#             best_idxs = good_idxs
#     return best_coeffs, best_idxs
#
#
# # Now go through and find lines of best fit using RANSAC
# # After each line has been found, mask off the points that
# #    were used to create it
# line_coeffs = []
# completed_points_mask = np.zeros_like(locs, dtype=np.uint8)
# locs_masked = np.ma.masked_array(locs, mask=completed_points_mask)
#
# num_lines_to_find = 40
# n_cols = 4
# n_rows = 10 # 3
#
# for line_idx in range(num_lines_to_find):
#     coeffs, idxs = RANSAC_LINEST(locs_masked, max_deviation=5, n_points=2)
#     line_coeffs.append(coeffs)
#
#     completed_points_mask[idxs, :] = 1
#     locs_masked.mask = completed_points_mask
#
#     x_lim = (locs[idxs, 0].min(), locs[idxs, 0].max())
#     xs = np.arange(x_lim[0], x_lim[1], 1)
#     ys = xs * coeffs[0] + coeffs[1]
#     plt.subplot(n_rows, n_cols, line_idx + 1)
#     plt.title("Estimate for line {}".format(line_idx))
#     plt.imshow(image)
#     plt.plot(xs, ys)
# plt.show()
#

#

# ###################   Methode 4 : lignes en combinaison avec méthode 2 (fonctionne à peu pres)  ##############################
#
# import matplotlib.pyplot as plt
#
# # Load the image in, convert to black and white
# image = image_gray  # image_mask
# # _, image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY_INV)
# plt.subplot(2, 2, 1)
# plt.imshow(image, cmap="gray")
# plt.title("Back and White Source Image")
#
# image_to_compute = image  # only do this to make experimenting with commenting out parts later
#
# # Get the derivatives
# # For some reason taking the x and y at the same time did not work right
# # This may not be necassary in your case, if there is a lot of noise in the image
# #   it may help
# # Comment out the sobel operations and the morphology and see what happens
# # Sobel of image
# img_sobel_x = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3)
# img_sobel_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3)
# img_sobel = np.abs(img_sobel_x) + np.abs(img_sobel_y)
# _, image_to_compute = cv2.threshold(img_sobel, .001, 1, cv2.THRESH_BINARY)
# plt.subplot(2, 2, 2)
# plt.imshow(image_to_compute, cmap="gray")
# plt.title("Thresholded Sobel of Input")
# # Morphology on image
# # again, this part may not be needed, but will help with noisy images
# kernel = np.ones((9, 9), np.uint8)
# closing = cv2.morphologyEx(image_to_compute, cv2.MORPH_CLOSE, kernel)
# image_to_compute = (closing * 255).astype(np.uint8)
# plt.subplot(2, 2, 3)
# plt.imshow(image_to_compute, cmap="gray")
# plt.title("Morphlogy on Image")
#
#
# # Get the lines as 1 pixel wide using skeletonization
# # Making all lines skinny makes the compute go faster
# def skeletonize(img):
#     '''
#     https://medium.com/analytics-vidhya/skeletonization-in-python-using-opencv-b7fa16867331
#     Get all the lines as 1 pixel wide
#     '''
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     skel = np.zeros(img.shape, np.uint8)
#     # Repeat steps 2-4
#     while True:
#         # Step 2: Open the image
#         opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
#         # Step 3: Substract open from the original image
#         temp = cv2.subtract(img, opening)
#         # Step 4: Erode the original image and refine the skeleton
#         eroded = cv2.erode(img, element)
#         skel = cv2.bitwise_or(skel, temp)
#         img = eroded.copy()
#         # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
#         if cv2.countNonZero(img) == 0:
#             break
#     return skel
#
#
# skel = skeletonize(image_to_compute)
# plt.subplot(2, 2, 4)
# plt.imshow(skel, cmap="gray")
# plt.title("Skeletonized Image")
#
# plt.show()
#
# # Now get the indicies where the pixels have a line in them
# y, x = np.where(skel > 0)
# locs = np.array((x, y), dtype=np.float32).T
# print("Have {} pixels to find lines through".format(locs.shape))
#
#
# # Use a RANSAC algorithm to compute the lines
# # This will randomly sample points, compute the line through them
# #    then find how many pixels are within the max_devaition parameter
# # The more points within the max_deviation parameter the better the line fits
# def RANSAC_LINEST(points, max_deviation=2., n_points=3):
#     '''
#     Approximate a line from the data
#     Returns the (slope,y_intercept) of the line that has the most points within max_deviation of the line
#     '''
#
#     mask_probabilities = ~points[:, 0].mask / points[:, 0].count()
#     best_coeffs = [0, 0]
#     best_count = 0
#     best_idxs = []
#     for it in range(100):
#         # choose random points, and compute the line of best fit for them
#         idxs = np.random.choice(points.shape[0], size=n_points, p=mask_probabilities, replace=False)
#         pts = points[idxs, :]
#         poly = np.poly1d(np.polyfit(pts[:, 0], pts[:, 1], deg=1))
#
#         # compute the resulting points and find matches
#         computed_ys = poly(points[:, 0])
#         deltas = np.abs(points[:, 1] - computed_ys)
#         good_idxs = np.where(deltas < max_deviation)[0]
#         if len(good_idxs) > best_count:
#             best_count = len(good_idxs)
#             best_coeffs = poly.coefficients
#             best_idxs = good_idxs
#     return best_coeffs, best_idxs
#
#
# # Now go through and find lines of best fit using RANSAC
# # After each line has been found, mask off the points that
# #    were used to create it
# line_coeffs = []
# completed_points_mask = np.zeros_like(locs, dtype=np.uint8)
# locs_masked = np.ma.masked_array(locs, mask=completed_points_mask)
#
# num_lines_to_find = 5
# n_cols = 2
# n_rows = 3
#
# for line_idx in range(num_lines_to_find):
#     coeffs, idxs = RANSAC_LINEST(locs_masked, max_deviation=5, n_points=2)
#     line_coeffs.append(coeffs)
#
#     completed_points_mask[idxs, :] = 1
#     locs_masked.mask = completed_points_mask
#
#     x_lim = (locs[idxs, 0].min(), locs[idxs, 0].max())
#     xs = np.arange(x_lim[0], x_lim[1], 1)
#     ys = xs * coeffs[0] + coeffs[1]
#     plt.subplot(n_rows, n_cols, line_idx + 1)
#     plt.title("Estimate for line {}".format(line_idx))
#     plt.imshow(image)
#     plt.plot(xs, ys)
# plt.show()
#




###################   Methode 5 : lignes en combinaison avec méthode 2 ##############################

filter = False

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image_gray, 100, 200, apertureSize=3)
kernel = np.ones((3, 3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
kernel = np.ones((5, 5), np.uint8)
edges = cv2.erode(edges, kernel, iterations=1)
# cv2.imwrite('canny.jpg', edges)

lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

if not lines.any():
    print('No lines were found')
    exit()

if filter:
    rho_threshold = 15
    theta_threshold = 0.1

    # how many lines are similar to a given one
    similar_lines = {i : [] for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue

            rho_i, theta_i = lines[i][0]
            rho_j, theta_j = lines[j][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                similar_lines[i].append(j)

    # ordering the INDECES of the lines by how many are similar to them
    indices = [i for i in range(len(lines))]
    indices.sort(key=lambda x : len(similar_lines[x]))

    # line flags is the base for the filtering
    line_flags = len(lines)*[True]
    for i in range(len(lines) - 1):
        if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
            continue

        for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
            if not line_flags[indices[j]]: # and only if we have not disregarded them already
                continue

            rho_i, theta_i = lines[indices[i]][0]
            rho_j, theta_j = lines[indices[j]][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

print('number of Hough lines:', len(lines))

filtered_lines = []

if filter:
    for i in range(len(lines)): # filtering
        if line_flags[i]:
            filtered_lines.append(lines[i])

    print('Number of filtered lines:', len(filtered_lines))
else:
    filtered_lines = lines

for line in filtered_lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)

    cv2.line(image_gray, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('hough.jpg', img)