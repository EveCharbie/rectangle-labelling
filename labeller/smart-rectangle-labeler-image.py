
import cv2
import numpy as np
from tqdm.notebook import tqdm
import pickle
import seaborn as sns
import scipy
import matplotlib.pyplot as plt



def draw_points_and_lines():
    global points_labels, circle_colors, circle_radius, small_image, active_points, rectangle_color, number_of_points_to_label

    for i in range(number_of_points_to_label):
        if active_points[i] == True:
            mouse_click_position = (int(points_labels[label_keys[i]][0]), int(points_labels[label_keys[i]][1]))
            cv2.circle(small_image, mouse_click_position, circle_radius, color=circle_colors[i], thickness=-1)
            for j in neighbors[i]:
                if active_points[j] == True:
                    line_position = (int(points_labels[label_keys[j]][0]), int(points_labels[label_keys[j]][1]))
                    cv2.line(small_image, mouse_click_position, line_position, line_color, thickness=1)

    cv2.imshow(Image_name, small_image)
    return

def xy_to_rtheta(x0, x1, y0, y1):
    if (y1 - y0) == 0:
        theta = np.pi/2
    else:
        theta = np.arctan((x0 - x1) / (y1 - y0))
    r = x0 * np.cos(theta) + y0 * np.sin(theta)
    return (r, theta)


def image_treatment(*args):
    global label_keys, number_of_points_to_label, points_labels, small_image

    last_frame_points_labels = np.zeros((2, number_of_points_to_label))
    for i, key in enumerate(label_keys):
        last_frame_points_labels[:, i] = points_labels[key]

    lines_last_frame, borders_last_frame, lines_points_index, borders_points_index, unique_lines_index, unique_borders_index = find_lines_to_search_for(last_frame_points_labels)

    print("lines_last_frame : ", lines_last_frame)
    print('borders_last_frame : ', borders_last_frame)


    # plt.figure()
    # plt.imshow(small_image)
    # for i in range(len(lines_last_frame)):
    #     plt.plot(np.array([lines_last_frame[i][0], lines_last_frame[i][1]]),
    #              np.array([lines_last_frame[i][2], lines_last_frame[i][3]]), '-r')
    # for i in range(len(borders_last_frame)):
    #     plt.plot(np.array([borders_last_frame[i][0], borders_last_frame[i][1]]),
    #              np.array([borders_last_frame[i][2], borders_last_frame[i][3]]), '-k')
    # plt.show()

    if len(unique_lines_index) > 0 or len(unique_borders_index) > 0:
        lines_new_vert, lines_new_horz, lines_new_vert_index, lines_new_horz_index = find_lines_through_pixels(lines_last_frame, borders_last_frame, lines_points_index, borders_points_index, unique_lines_index, unique_borders_index)
        points_new = find_points_next_frame(lines_new_vert, lines_new_horz, lines_new_vert_index, lines_new_horz_index)
        fixation_pixel = distort_to_rectangle(points_new, lines_new_vert_index, lines_new_horz_index)
    # ajouter au graph de taille definie + tous les rectangles
    # save fixation pixel
    # save figure distorted ?

    return

def find_lines_to_search_for(last_frame_points_labels):
    global active_points, number_of_points_to_label, borders_points
    threashold_r = 8
    threashold_theta = 0.1

    lines = []
    lines_rtheta = []
    lines_points_index = []
    borders = []
    borders_rtheta = []
    borders_points_index = []
    lines_index = []
    borders_index = []
    for i in range(number_of_points_to_label):
        if active_points[i] == True:
            x0 = int(last_frame_points_labels[0, i])
            y0 = int(last_frame_points_labels[1, i])
            for j in neighbors[i]:
                if j > i:
                    if active_points[j] == True:
                        x1 = int(last_frame_points_labels[0, j])
                        y1 = int(last_frame_points_labels[1, j])
                        if [i, j] in borders_pairs or [j, i] in borders_pairs:
                            print(f"Borders : {i} {j}")
                            borders += [np.array([x0, x1, y0, y1])]
                            borders_rtheta += [xy_to_rtheta(x0, x1, y0, y1)]
                            borders_points_index += [[i, j]]
                            borders_index += [lines_definitions[i, j]]
                        else:
                            print(f"Lines : {i} {j}")
                            lines += [np.array([x0, x1, y0, y1])]
                            lines_rtheta += [xy_to_rtheta(x0, x1, y0, y1)]
                            lines_points_index += [[i, j]]
                            lines_index += [lines_definitions[i, j]]
    unique_lines = []
    unique_lines_rtheta = []
    unique_lines_points_index = []
    unique_lines_index = []
    for i in range(len(lines)):
        if not unique_lines:
            unique_lines = [lines[0]]
            unique_lines_rtheta = [lines_rtheta[0]]
            unique_lines_points_index += [lines_points_index[i]]
            unique_lines_index += [lines_index[i]]
        else:
            is_unique = True
            for j in range(len(unique_lines)):
                if abs(lines_rtheta[i][0] - lines_rtheta[j][0]) < threashold_r and abs(lines_rtheta[i][1] - lines_rtheta[j][1]) < threashold_theta:
                    is_unique = False
            if is_unique:
                unique_lines += [lines[i]]
                unique_lines_rtheta += [lines_rtheta[i]]
                unique_lines_points_index += [lines_points_index[i]]
                unique_lines_index += [lines_index[i]]

    unique_borders = []
    unique_borders_rtheta = []
    unique_borders_points_index = []
    unique_borders_index = []
    for i in range(len(borders)):
        if not unique_borders:
            unique_borders = [borders[0]]
            unique_borders_rtheta = [borders_rtheta[0]]
            unique_borders_points_index += [borders_points_index[i]]
            unique_borders_index += [borders_index[i]]
        else:
            is_unique = True
            for j in range(len(unique_borders)):
                if abs(borders_rtheta[i][0] - borders_rtheta[j][0]) < threashold_r and abs(borders_rtheta[i][1] - borders_rtheta[j][1]) < threashold_theta:
                    is_unique = False
            if is_unique:
                unique_borders += [borders[i]]
                unique_borders_rtheta += [borders_rtheta[i]]
                unique_borders_points_index += [borders_points_index[i]]
                unique_borders_index += [borders_index[i]]

    return unique_lines, unique_borders, unique_lines_points_index, unique_borders_points_index, unique_lines_index, unique_borders_index

def find_line_pixels():
    # lower = np.array([0, 25, 108], dtype="uint8")
    # upper = np.array([26, 82, 136], dtype="uint8")
    lower = np.array([0, 0, 115], dtype="uint8")
    upper = np.array([104, 32, 171], dtype="uint8")
    hsv = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    image_mask = cv2.bitwise_and(hsv, hsv, mask=mask)
    image_gray = cv2.cvtColor(image_mask, cv2.COLOR_RGB2GRAY)
    return image_gray

def find_border_pixels():

    # lower = np.array([0, 0, 150], dtype="uint8")
    # upper = np.array([179, 229, 232], dtype="uint8")
    lower = np.array([98, 128, 0], dtype="uint8")
    upper = np.array([179, 255, 255], dtype="uint8")
    hsv = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    image_mask = cv2.bitwise_and(hsv, hsv, mask=mask)
    image_gray_very_small = cv2.cvtColor(image_mask, cv2.COLOR_RGB2GRAY)

    # very_small_image = small_image
    # image_gray_very_small = cv2.cvtColor(very_small_image, cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(image_gray_very_small, 0, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL
    # plt.figure()
    # plt.imshow(image_gray_very_small)

    image_output = np.zeros(np.shape(image_gray_very_small))

    for cnt in contours:
        # plt.plot(cnt[:, :, 0], cnt[:, :, 1], '.k')
        for i in range(len(cnt)):
            image_output[cnt[i, :, 1], cnt[i, :, 0]] = 1
    # plt.show()

    # plt.figure()
    # plt.imshow(image_output)
    # plt.show()

    return image_output

def find_lines_through_pixels(lines_last_frame, borders_last_frame, lines_points_index, borders_points_index, unique_lines_index, unique_borders_index):

    image_gray_lines = find_line_pixels()
    image_gray_borders = find_border_pixels()

    # plt.figure()
    # plt.imshow(small_image)

    coeffs = {"a": [], "b": []}
    lines_new_vert = np.array([])
    lines_new_horz = np.array([])
    lines_new_vert_index = []
    lines_new_horz_index = []

    idx_points_in_rectangle_lines = find_pixel_close_to_lines(lines_last_frame, image_gray_lines)
    idx_points_in_rectangle_borders = find_pixel_close_to_lines(borders_last_frame, image_gray_borders)

    idx_points_in_rectangle = idx_points_in_rectangle_lines + idx_points_in_rectangle_borders
    lines_borders_last_frame = lines_last_frame + borders_last_frame
    lines_borders_points_index = lines_points_index + borders_points_index
    lines_borders_index = unique_lines_index + unique_borders_index

    for i, line in enumerate(lines_borders_last_frame):

        pts = idx_points_in_rectangle[i]
        if np.any(pts > 0):
            y, x = np.where(pts > 0)
            locs = np.array((x, y), dtype=np.float32)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(locs)
            if np.isnan(slope):
                print('PROBLEME slope = nan')
                continue
            coeffs["a"].append(slope)
            coeffs["b"].append(intercept)
            x_lim = (locs[0, :].min(), locs[0, :].max())
            xs = np.arange(x_lim[0], x_lim[1], 1)
            ys = xs * coeffs["a"][-1] + coeffs["b"][-1]
            # plt.plot(xs, coeffs["a"][-1] * xs + coeffs["b"][-1], '-b')
            line_rtheta_new = xy_to_rtheta(xs[0], xs[-1], ys[0], ys[-1])

            present_pair_reversed = [0, 0]
            present_pair_reversed[:] = lines_borders_points_index[i][:]
            present_pair_reversed.reverse()

            if lines_borders_points_index[i] in vert_pairs or present_pair_reversed in vert_pairs:
                if len(lines_new_vert) == 0:
                    lines_new_vert = np.array([line_rtheta_new[0], line_rtheta_new[1]])
                else:
                    lines_new_vert = np.vstack((lines_new_vert, np.array([line_rtheta_new[0], line_rtheta_new[1]])))
                lines_new_vert_index += [lines_borders_index[i]]
            elif lines_borders_points_index[i] in horz_pairs or present_pair_reversed in horz_pairs:
                if len(lines_new_horz) == 0:
                    lines_new_horz = np.array([line_rtheta_new[0], line_rtheta_new[1]])
                else:
                    lines_new_horz = np.vstack((lines_new_horz, np.array([line_rtheta_new[0], line_rtheta_new[1]])))
                lines_new_horz_index += [lines_borders_index[i]]
        else:
            print('PROBLEME, la sélection par couleurs ne fonctionne pas !')
            continue

    # plt.show()

    return lines_new_vert, lines_new_horz, lines_new_vert_index, lines_new_horz_index

def find_pixel_close_to_lines(lines, image_gray):
    slack = 5
    pixels_close = [np.zeros((np.shape(image_gray)[0], np.shape(image_gray)[1])) for i in range(len(lines))]

    plt.figure()
    plt.imshow(image_gray)
    for ix in range(np.shape(image_gray)[1]):
        print(f'Pixels X {ix}')
        for iy in range(np.shape(image_gray)[0]):
            for iline, line in enumerate(lines):
                vect_line = np.array([line[1], line[3]]) - np.array([line[0], line[2]])
                vect_to_point = np.array([ix, iy]) - np.array([line[0], line[2]])
                distance = np.abs(np.cross(vect_line, vect_to_point)) / np.linalg.norm(vect_line)
                if image_gray[iy, ix] > 0:
                    plt.plot(np.array([ix]), np.array([iy]), 'ow')
                if distance < slack:
                    plt.plot(np.array([ix]), np.array([iy]), '.y')
                if image_gray[iy, ix] > 0 and distance < slack:
                    pixels_close[iline][iy, ix] = 1
                    plt.plot(np.array([ix]), np.array([iy]), '.m')
    plt.show()

    # fig, ax = plt.subplots(3, 4)
    # axs = ax.ravel()
    # for iline, line in enumerate(lines):
    #     axs[iline].imshow(image_gray)
    #     y, x = np.where(pixels_close[iline] > 0)
    #     locs = np.array((x, y), dtype=np.float32)
    #     axs[iline].plot(locs[0], locs[1], '.m')
    #     axs[iline].plot(np.array([line[0], line[1]]), np.array([line[2], line[3]]), '-.r')
    # plt.show()

    return pixels_close

def find_points_next_frame(lines_new_vert, lines_new_horz, lines_new_vert_index, lines_new_horz_index):

    def intersection(line1, line2):
        """Finds the intersection of two lines given in Hesse normal form.
        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = line1
        rho2, theta2 = line2
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]

    # image_lines_points = small_image.copy()
    if len(np.shape(lines_new_vert)) > 1:
        lines_new_vert_order = np.argsort(np.abs(lines_new_vert[:, 0]))
        lines_new_vert = lines_new_vert[lines_new_vert_order]

    if len(np.shape(lines_new_horz)) > 1:
        lines_new_horz_order = np.argsort(np.abs(lines_new_horz[:, 0]))
        lines_new_horz = lines_new_horz[lines_new_horz_order]

    # Finding the intersection points of the lines
    points = []
    num_lines_new_vert = lines_new_vert.shape[0]
    num_lines_new_horz = lines_new_horz.shape[0]
    active_points[:] = False
    for i in range(num_lines_new_vert):
        for j in range(num_lines_new_horz):
            point = intersection(lines_new_vert[i], lines_new_horz[j])
            points.append(point)
            point_index = int(points_definitions[int(lines_new_vert_index[i]), int(lines_new_horz_index[j])])
            points_labels[str(point_index)] = point[0]
            active_points[point_index] = True

    # plt.figure()
    # plt.imshow(small_image)

    # Drawing the lines and points
    lines_to_plot = []
    lines = np.vstack((lines_new_horz, lines_new_vert))
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # cv2.line(image_lines_points, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # plt.plot(np.array([x1, x2]), np.array([y1, y2]), '-b')
        # lines_to_plot += [(x1, x2, y1, y2)]

    # for p in points:
    #     cv2.circle(image_lines_points, (p[0][0], p[0][1]), 5, 255, 2)
    #     plt.plot(p[0][0], p[0][1], '.g')

    # Displaying the results
    # cv2.imshow(Image_name_approx, image_lines_points)
    # cv2.waitKey(0)
    # plt.show()

    return points


def distort_to_rectangle(points, lines_new_vert_index, lines_new_horz_index):
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(image_to_distort, four_vertices_transform, position_corners_to_map):
        wraped_width = int(abs(position_corners_to_map[0, 0] - position_corners_to_map[1, 0]))
        wraped_height = int(abs(position_corners_to_map[0, 1] - position_corners_to_map[3, 1]))
        dst = np.array([
            [position_corners_to_map[0, 0], position_corners_to_map[0, 1]],
            [position_corners_to_map[1, 0], position_corners_to_map[1, 1]],
            [position_corners_to_map[2, 0], position_corners_to_map[2, 1]],
            [position_corners_to_map[3, 0], position_corners_to_map[3, 1]]], dtype="float32")
        M = cv2.getPerspectiveTransform(four_vertices_transform, dst)
        wraped = cv2.warpPerspective(image_to_distort, M, (wraped_width, wraped_height))
        wraped = np.round(wraped)
        wraped = wraped.astype(np.uint8)
        return wraped

    def which_rectangle_is_visible(lines_new_vert_index, lines_new_horz_index):

        for i in range(len(rectangle_lines_definitions)):
            sum_rectangle_lines = 0
            for j in range(4):
                if rectangle_lines_definitions[i][j] in lines_new_vert_index or rectangle_lines_definitions[i][j] in lines_new_horz_index:
                    sum_rectangle_lines += 1
            if sum_rectangle_lines == 4:
                break
        if i == len(rectangle_lines_definitions)-1 and sum_rectangle_lines == 4:
            position_corners_to_map = None
        else:
            position_corners_to_map = rectangle_points_definitions[i]

        pts = np.zeros((4, 2), dtype="float32")
        four_vertices = order_points(pts)

        return position_corners_to_map, four_vertices

    # Finding the four corner points and ordering them
    pts = np.asarray(points)
    four_vertices = order_points(pts.reshape(pts.shape[0], pts.shape[2]))

    min_left = np.min(four_vertices[0, :])
    max_right = np.max(four_vertices[0, :])
    min_top = np.min(four_vertices[1, :])
    max_bottom = np.max(four_vertices[1, :])

    image_to_distort = small_image.copy()

    four_vertices_transform = np.zeros((4, 2))
    four_vertices_transform[:, :] = four_vertices[:, :]
    size_width = width_small
    size_height = height_small
    if min_left < 0:
        missing_pixels = int(abs(min_left))
        zeros_left = np.ones((missing_pixels, size_width, 3)) * 0.5
        image_to_distort = np.hstack((zeros_left, image_to_distort))
        size_height += missing_pixels
        four_vertices_transform[:, 0] += missing_pixels
    if max_right > width_small:
        missing_pixels = int(max_right - width_small)
        zeros_right = np.ones((missing_pixels, size_width, 3)) * 0.5
        image_to_distort = np.hstack((image_to_distort, zeros_right))
        size_height += missing_pixels
    if min_top < 0:
        missing_pixels = int(abs(min_top))
        zeros_top = np.ones((missing_pixels, size_height, 3)) * 0.5
        image_to_distort = np.vstack((zeros_top, image_to_distort))
        size_width += missing_pixels
        four_vertices_transform[:, 1] += missing_pixels
    if max_bottom > height_small:
        missing_pixels = int(max_bottom - height_small)
        zeros_bottom = np.ones((missing_pixels, size_height, 3)) * 0.5
        image_to_distort = np.vstack((image_to_distort, zeros_bottom))
        size_width += missing_pixels

    # Perspective transform to get the warped image
    four_vertices_transform = four_vertices_transform.astype(np.float32)
    position_corners_to_map = which_rectangle_is_visible(lines_new_vert_index, lines_new_horz_index)
    wraped = four_point_transform(image_to_distort, four_vertices_transform, position_corners_to_map)


    plt.figure()
    image_to_distort = image_to_distort.astype(np.uint8)
    plt.imshow(image_to_distort)
    for i in range(4):
        plt.plot(four_vertices_transform[i, 0], four_vertices_transform[i, 1], '.r')
    plt.show()
    #
    # # Displaying the results
    plt.figure()
    plt.imshow(wraped)
    plt.show()

    mask = cv2.inRange(wraped, (0, 255, 0), (0, 255, 0))
    if len(np.where(mask != 0)[0]) > 0:
        fixation_region_y, fixation_region_x = np.where(mask != 0)
        fixation_pixel = np.round(np.array([np.mean(fixation_region_x), np.mean(fixation_region_y)]))
        fixation_pixel = fixation_pixel.astype(int)
        cv2.circle(wraped, (fixation_pixel[0], fixation_pixel[1]), 1, color=(0, 255, 255), thickness=-1)
    else:
        fixation_pixel = None

    cv2.line(wraped, (53, 0), (53, 428), (0, 0, 0), 2)
    cv2.line(wraped, (161, 0), (161, 428), (0, 0, 0), 2)
    cv2.line(wraped, (0, 107), (214, 107), (0, 0, 0), 2)
    cv2.line(wraped, (0, 322), (214, 322), (0, 0, 0), 2)
    cv2.line(wraped, (53, 160), (161, 160), (0, 0, 0), 2)
    cv2.line(wraped, (53, 268), (161, 268), (0, 0, 0), 2)
    cv2.imshow("Distorted", wraped)

    return fixation_pixel


def mouse_click(event, x, y, flags, param):
    global points_labels, current_click

    if event == cv2.EVENT_LBUTTONDOWN:
        points_labels[label_keys[current_click]] = np.array([x, y])
        draw_points_and_lines()
    return

def point_choice(*args):
    global current_click
    num_point = args[1]
    current_click = num_point
    if active_points[num_point]:
        active_points[num_point] = False
    else:
        active_points[num_point] = True
    draw_points_and_lines()
    return

############################### code beginning #######################################################################
global small_image, number_of_points_to_label, width_small, height_small, label_keys, points_labels, frames_clone
global ratio_image, Image_name, borders_points, borders_pairs, fixation, rectangle_lines_definitions

circle_radius = 5
line_color = (1, 1, 1)
number_of_points_to_label = 20
circle_colors = sns.color_palette(palette="viridis", n_colors=number_of_points_to_label)
for i in range(number_of_points_to_label):
    col_0 = int(circle_colors[i][0] * 255)
    col_1 = int(circle_colors[i][1] * 255)
    col_2 = int(circle_colors[i][2] * 255)
    circle_colors[i] = (col_0, col_1, col_2)

Image_name = "Video"
Image_name_approx = "Video_approx"
Trackbar_name = "Frames"
ratio_image = 5

file = open(f"../output/PI world v1 ps1_181_undistorted_images.pkl", "rb")
image = pickle.load(file)
fixation_small = np.array([105, 108])

num_frames = 1

points_labels = {"0": np.zeros((2, )),
                "1": np.zeros((2, )),
                "2": np.zeros((2, )),
                "3": np.zeros((2, )),
                "4": np.zeros((2, )),
                "5": np.zeros((2, )),
                "6": np.zeros((2, )),
                "7": np.zeros((2, )),
                "8": np.zeros((2, )),
                "9": np.zeros((2, )),
                "10": np.zeros((2, )),
                "11": np.zeros((2, )),
                "12": np.zeros((2, )),
                "13": np.zeros((2, )),
                "14": np.zeros((2, )),
                "15": np.zeros((2, )),
                "16": np.zeros((2, )),
                "17": np.zeros((2, )),
                "18": np.zeros((2, )),
                "19": np.zeros((2, ))}
label_keys = [key for key in points_labels.keys()]
current_click = 0
active_points = np.zeros((number_of_points_to_label,))
neighbors = [[1, 2, 3, 4, 14, 16],  # 0
             [0, 2, 3, 6, 12, 17],  # 1
             [0, 1, 3, 7, 13, 18],  # 2
             [0, 1, 2, 5, 15, 19],  # 3
             [5, 0, 14, 16],  # 4
             [4, 3, 15, 19],  # 5
             [7, 1, 12, 17],  # 6
             [6, 2, 13, 18],  # 7
             [11],  # 8
             [10],  # 9
             [9],  # 10
             [8],  # 11
             [13, 1, 6, 17],  # 12
             [12, 2, 7, 18],  # 13
             [15, 0, 4, 16],  # 14
             [14, 3, 5, 19],  # 15
             [17, 18, 19, 0, 4, 14],  # 16
             [16, 18, 19, 1, 6, 12],  # 17
             [16, 17, 19, 2, 7, 13],  # 18
             [16, 17, 18, 3, 5, 15]] # 19

vert_pairs = [[0, 4],
             [0, 14],
             [0, 16],
             [4, 14],
             [4, 16],
             [14, 16],
             [1, 6],
             [1, 12],
             [1, 17],
             [6, 12],
             [6, 17],
             [12, 17],
             [8, 11],
             [2, 7],
             [2, 13],
             [2, 18],
             [7, 13],
             [7, 18],
             [13, 18],
             [3, 5],
             [3, 15],
             [3, 19],
             [5, 15],
             [5, 19],
             [15, 19]]

horz_pairs = [[0, 1],
             [0, 2],
             [0, 3],
             [1, 2],
             [1, 3],
             [2, 3],
             [4, 5],
             [6, 7],
             [9, 10],
             [12, 13],
             [14, 15],
             [16, 17],
             [16, 18],
             [16, 19],
             [17, 18],
             [17, 19],
             [18, 19]]

borders_points = [0, 3, 16, 19]
borders_lines = [0, 1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19]

borders_pairs = [[0, 1],
                 [0, 2],
                 [0, 3],
                 [0, 4],
                 [0, 14],
                 [0, 16],
                 [4, 14],
                 [4, 16],
                 [14, 16],
                 [1, 2],
                 [1, 3],
                 [2, 3],
                 [3, 5],
                 [3, 15],
                 [3, 19],
                 [5, 15],
                 [5, 19],
                 [15, 19],
                 [16, 17],
                 [16, 18],
                 [16, 19],
                 [17, 18],
                 [17, 19],
                 [18, 19]]

lines_definitions = np.zeros((20, 20))
lines_definitions[:, :] = np.nan
lines_definitions[0, 1:4] = 0
lines_definitions[1:4, 0] = 0
lines_definitions[1, [0, 2, 3]] = 0
lines_definitions[[0, 2, 3], 1] = 0
lines_definitions[2, [0, 1, 3]] = 0
lines_definitions[[0, 1, 3], 2] = 0
lines_definitions[3, :3] = 0
lines_definitions[:3, 3] = 0
lines_definitions[4, 5] = 1
lines_definitions[5, 4] = 1
lines_definitions[6, 7] = 2
lines_definitions[7, 6] = 2
lines_definitions[9, 10] = 3
lines_definitions[10, 9] = 3
lines_definitions[12, 13] = 4
lines_definitions[13, 12] = 4
lines_definitions[14, 15] = 5
lines_definitions[15, 14] = 5
lines_definitions[16, 17:20] = 6
lines_definitions[17:20, 16] = 6
lines_definitions[17, [16, 18, 19]] = 6
lines_definitions[[16, 18, 19], 17] = 6
lines_definitions[18, [16, 17, 19]] = 6
lines_definitions[[16, 17, 19], 18] = 6
lines_definitions[19, 16:19] = 6
lines_definitions[16:19, 19] = 6
lines_definitions[0, [4, 14, 16]] = 7
lines_definitions[[4, 14, 16], 0] = 7
lines_definitions[4, [0, 14, 16]] = 7
lines_definitions[[0, 14, 16], 4] = 7
lines_definitions[14, [0, 4, 16]] = 7
lines_definitions[[0, 4, 16], 14] = 7
lines_definitions[16, [0, 4, 14]] = 7
lines_definitions[[0, 4, 14], 16] = 7
lines_definitions[1, [6, 12, 17]] = 8
lines_definitions[[6, 12, 17], 1] = 8
lines_definitions[6, [1, 12, 17]] = 8
lines_definitions[[1, 12, 17], 6] = 8
lines_definitions[12, [1, 6, 17]] = 8
lines_definitions[[1, 6, 17], 12] = 8
lines_definitions[17, [1, 6, 12]] = 8
lines_definitions[[1, 6, 12], 17] = 8
lines_definitions[8, 11] = 9
lines_definitions[11, 8] = 9
lines_definitions[2, [7, 13, 18]] = 10
lines_definitions[[7, 13, 18], 2] = 10
lines_definitions[7, [2, 13, 18]] = 10
lines_definitions[[2, 13, 18], 7] = 10
lines_definitions[13, [2, 7, 18]] = 10
lines_definitions[[2, 7, 18], 13] = 10
lines_definitions[18, [2, 7, 13]] = 10
lines_definitions[[2, 7, 13], 18] = 10
lines_definitions[3, [5, 15, 19]] = 11
lines_definitions[[5, 15, 19], 3] = 11
lines_definitions[5, [3, 15, 19]] = 11
lines_definitions[[3, 15, 19], 5] = 11
lines_definitions[15, [3, 5, 19]] = 11
lines_definitions[[3, 5, 19], 15] = 11
lines_definitions[19, [3, 5, 15]] = 11
lines_definitions[[3, 5, 15], 19] = 11

points_definitions = np.zeros((12, 12))
points_definitions[:, :] = np.nan
points_definitions[0, 7] = 0
points_definitions[7, 0] = 0
points_definitions[0, 8] = 1
points_definitions[8, 0] = 1
points_definitions[0, 10] = 2
points_definitions[10, 0] = 2
points_definitions[0, 11] = 3
points_definitions[11, 0] = 3
points_definitions[1, 7] = 4
points_definitions[7, 1] = 4
points_definitions[1, 11] = 5
points_definitions[11, 1] = 5
points_definitions[2, 8] = 6
points_definitions[8, 2] = 6
points_definitions[2, 10] = 7
points_definitions[10, 2] = 7
points_definitions[4, 8] = 12
points_definitions[8, 4] = 12
points_definitions[4, 10] = 13
points_definitions[10, 4] = 13
points_definitions[5, 7] = 14
points_definitions[7, 5] = 14
points_definitions[5, 11] = 15
points_definitions[11, 5] = 15
points_definitions[6, 7] = 16
points_definitions[7, 6] = 16
points_definitions[6, 8] = 17
points_definitions[8, 6] = 17
points_definitions[6, 10] = 18
points_definitions[10, 6] = 18
points_definitions[6, 11] = 19
points_definitions[11, 6] = 19

rectangle_lines_definitions = np.zeros((11, 4))
rectangle_lines_definitions[0, :] = np.array([0, 3, 19, 16])
rectangle_lines_definitions[1, :] = np.array([0, 3, 15, 14])
rectangle_lines_definitions[2, :] = np.array([4, 5, 19, 16])
rectangle_lines_definitions[3, :] = np.array([1, 2, 19, 17])
rectangle_lines_definitions[4, :] = np.array([0, 2, 18, 16])
rectangle_lines_definitions[5, :] = np.array([4, 5, 15, 14])
rectangle_lines_definitions[6, :] = np.array([0, 3, 5, 4])
rectangle_lines_definitions[7, :] = np.array([14, 15, 19, 16])
rectangle_lines_definitions[8, :] = np.array([6, 7, 13, 12])
rectangle_lines_definitions[9, :] = np.array([0, 1, 17, 16])
rectangle_lines_definitions[10, :] = np.array([2, 3, 19, 18])

rectangle_points_definitions = np.zeros((11, 4))
rectangle_points_definitions[0, :] = np.array([0, 6, 7, 11])
rectangle_points_definitions[1, :] = np.array([0, 7, 11, 5])
rectangle_points_definitions[2, :] = np.array([1, 7, 11, 6])
rectangle_points_definitions[3, :] = np.array([0, 7, 10, 6])
rectangle_points_definitions[4, :] = np.array([0, 8, 11, 6])
rectangle_points_definitions[5, :] = np.array([1, 5, 7, 11])
rectangle_points_definitions[6, :] = np.array([0, 7, 11, 1])
rectangle_points_definitions[7, :] = np.array([5, 7, 11, 6])
rectangle_points_definitions[8, :] = np.array([2, 4, 8, 10])
rectangle_points_definitions[9, :] = np.array([0, 7, 8, 6])
rectangle_points_definitions[10, :] = np.array([0, 10, 11, 6])

rectangle_points_position_definition = np.zeros((11, 4, 2))
rectangle_points_position_definition[0, :, :] = np.array([[0, 0],
                                                 [214, 0],
                                                 [214, 428],
                                                 [0, 428]])
rectangle_points_position_definition[1, :, :] = np.array([[0, 0],
                                                 [214, 0],
                                                 [214, 322],
                                                 [0, 322]])
rectangle_points_position_definition[2, :, :] = np.array([[0, 107],
                                                 [214, 107],
                                                 [214, 428],
                                                 [0, 428]])
rectangle_points_position_definition[3, :, :] = np.array([[53, 0],
                                                 [214, 0],
                                                 [214, 428],
                                                 [53, 428]])
rectangle_points_position_definition[4, :, :] = np.array([[0, 0],
                                                 [161, 0],
                                                 [161, 428],
                                                 [0, 428]])
rectangle_points_position_definition[5, :, :] = np.array([[0, 107],
                                                 [214, 107],
                                                 [214, 322],
                                                 [0, 322]])
rectangle_points_position_definition[6, :, :] = np.array([[0, 0],
                                                 [214, 0],
                                                 [214, 107],
                                                 [0, 107]])
rectangle_points_position_definition[7, :, :] = np.array([[0, 322],
                                                 [214, 322],
                                                 [214, 428],
                                                 [0, 428]])
rectangle_points_position_definition[8, :, :] = np.array([[53, 160],
                                                 [161, 160],
                                                 [161, 268],
                                                 [52, 268]])
rectangle_points_position_definition[9, :, :] = np.array([[0, 0],
                                                 [53, 0],
                                                 [53, 428],
                                                 [0, 428]])
rectangle_points_position_definition[10, :, :] = np.array([[161, 0],
                                                  [214, 0],
                                                  [214, 428],
                                                  [0, 428]])





# points order on trampoline
# 0 - 1 - 2 - 3
# 4 - - - - - 5
# - - 6 - 7 - -
# - - - 8 - - -
# - - -910- - -
# - - - 11- - -
# - - 12-13 - -
# 14- - - - -15
# 16 -17-18- 19

def nothing(x):
    return

cv2.namedWindow(Image_name)
cv2.createButton("0", point_choice, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("1", point_choice, 1, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("2", point_choice, 2, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("3", point_choice, 3, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("4", point_choice, 4, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("5", point_choice, 5, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("6", point_choice, 6, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("7", point_choice, 7, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("8", point_choice, 8, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("9", point_choice, 9, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("10", point_choice, 10, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("11", point_choice, 11, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("12", point_choice, 12, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("13", point_choice, 13, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("14", point_choice, 14, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("15", point_choice, 15, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("16", point_choice, 16, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("17", point_choice, 17, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("18", point_choice, 18, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("19", point_choice, 19, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("OK",  image_treatment, 0, cv2.QT_PUSH_BUTTON, 0)
cv2.setMouseCallback(Image_name, mouse_click)

image_clone = image.copy()
width, height, rgb = np.shape(image_clone)
small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))
cv2.imshow(Image_name, small_image)
while True:
    image_clone = image.copy()
    width, height, rgb = np.shape(image_clone)
    small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))
    width_small, height_small, rgb_small = np.shape(small_image)
    # fixation_small = np.round(fixation/ratio_image)
    # fixation_small = fixation_small.astype(int)
    small_image[fixation_small[0]-1:fixation_small[0]+2, fixation_small[1]-1:fixation_small[1]+2, :] = np.array([0, 255, 0])
    # cv2.imshow(Image_name, small_image)
    key = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()

