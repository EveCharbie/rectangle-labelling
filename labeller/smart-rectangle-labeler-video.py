
import cv2
import numpy as np
from tqdm.notebook import tqdm
import pickle
import seaborn as sns


def load_video_frames(video_file, num_frames=None):
    video = cv2.VideoCapture(video_file)
    frames = []

    if num_frames is None:
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    num_frames_update = 0
    for i in tqdm(range(num_frames), desc='Loading video'):
        ret, frame = video.read()
        if type(frame) == np.ndarray:
            frames.append(frame)
            num_frames_update+=1

    video.release()

    return frames, num_frames_update


def draw_points_and_lines():
    global points_labels, circle_colors, circle_radius, frame_counter, small_image, active_points, rectangle_color, number_of_points_to_label

    for i in range(number_of_points_to_label):
        if active_points[frame_counter, i] == True:
            mouse_click_position = (int(points_labels[label_keys[i]][0, frame_counter]), int(points_labels[label_keys[i]][1, frame_counter]))
            cv2.circle(small_image, mouse_click_position, circle_radius, color=circle_colors[i], thickness=-1)
            for j in neighbors[i]:
                if active_points[frame_counter, j] == True:
                    line_position = (int(points_labels[label_keys[j]][0, frame_counter]), int(points_labels[label_keys[j]][1, frame_counter]))
                    cv2.line(small_image, mouse_click_position, line_position, line_color, thickness=1)

    cv2.imshow(Image_name, small_image)
    return


def mouse_click(event, x, y, flags, param):
    global points_labels, current_click, frame_counter

    if event == cv2.EVENT_LBUTTONDOWN:
        points_labels[label_keys[current_click]][:, frame_counter] = np.array([x, y])
        draw_points_and_lines()
    return


# 0 - 1 - 2 - 3
# 4 - - - - - 5
# - - 6 - 7 - -
# - - - 8 - - -
# - - -910- - -
# - - - 11- - -
# - - 12-13 - -
# 14- - - - -15
# 16 -17-18- 19


def point_choice(*args):
    global current_click
    num_point = args[1]
    current_click = num_point
    if active_points[frame_counter, num_point]:
        active_points[frame_counter, num_point] = False
    else:
        active_points[frame_counter, num_point] = True
    draw_points_and_lines()
    return

############################### code beginning #######################################################################
global small_image, image, image_bidon, number_of_points_to_label

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
Trackbar_name = "Frames"
ratio_image = 1.5

movie_path = "../input/"
movie_name = "PI world v1 ps1.mp4"
movie_file = movie_path + movie_name
frames, num_frames = load_video_frames(movie_file)
frames_clone = frames.copy()
points_labels = {"0": np.zeros((2, len(frames))),
                "1": np.zeros((2, len(frames))),
                "2": np.zeros((2, len(frames))),
                "3": np.zeros((2, len(frames))),
                "4": np.zeros((2, len(frames))),
                "5": np.zeros((2, len(frames))),
                "6": np.zeros((2, len(frames))),
                "7": np.zeros((2, len(frames))),
                "8": np.zeros((2, len(frames))),
                "9": np.zeros((2, len(frames))),
                "10": np.zeros((2, len(frames))),
                "11": np.zeros((2, len(frames))),
                "12": np.zeros((2, len(frames))),
                "13": np.zeros((2, len(frames))),
                "14": np.zeros((2, len(frames))),
                "15": np.zeros((2, len(frames))),
                "16": np.zeros((2, len(frames))),
                "17": np.zeros((2, len(frames))),
                "18": np.zeros((2, len(frames))),
                "19": np.zeros((2, len(frames)))}
label_keys = [key for key in points_labels.keys()]
current_click = 0
active_points = np.zeros((num_frames, number_of_points_to_label))
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

def nothing(x):
    return

cv2.namedWindow(Image_name)
cv2.createTrackbar('Frames', Image_name, 0, num_frames, nothing)
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
cv2.setMouseCallback(Image_name, mouse_click)

playVideo = True
frame_counter = 0
image_clone = frames[0].copy()
width, height, rgb = np.shape(image_clone)
small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))
cv2.imshow(Image_name, small_image)
while playVideo == True:

    key = cv2.waitKey(1) & 0xFF

    frame_counter = cv2.getTrackbarPos(Trackbar_name, Image_name)
    image_clone = frames_clone[frame_counter]
    small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))

    if key == ord(','):  # if `<` then go back
        print("frame_counter fleche -: ", frame_counter, ' -> ', frame_counter-1)
        frame_counter -= 1
        cv2.setTrackbarPos(Trackbar_name, Image_name, frame_counter)
        image_clone = frames_clone[frame_counter]
        small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))
        cv2.imshow(Image_name, small_image)
        draw_points_and_lines()

    elif key == ord('.'):  # if `>` then advance
        print("frame_counter fleche +: ", frame_counter, ' -> ', frame_counter+1)
        frame_counter += 1
        cv2.setTrackbarPos(Trackbar_name, Image_name, frame_counter)
        image_clone = frames_clone[frame_counter]
        small_image = cv2.resize(image_clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))
        cv2.imshow(Image_name, small_image)
        draw_points_and_lines()

    elif key == ord('x'):  # if `x` then quit
        playVideo = False

cv2.destroyAllWindows()

with open(f'../output/{movie_name[:-4]}_labeling_points.pkl', 'wb') as handle:
    pickle.dump([points_labels, active_points], handle)















