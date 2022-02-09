
import cv2
import numpy as np
from tqdm.notebook import tqdm
import pickle


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
    global points_labels, circle_colors, circle_radius, frame_counter, small_image, active_points, rectangle_color

    print("draw_points : ", frame_counter)
    for i in range(8):
        if active_points[frame_counter, i] == True:
            mouse_click_position = (int(points_labels[label_keys[i]][0, frame_counter]), int(points_labels[label_keys[i]][1, frame_counter]))
            cv2.circle(small_image, mouse_click_position, circle_radius, color=circle_colors[i], thickness=-1)
            for j in neighbors[i]:
                if active_points[frame_counter, j] == True:
                    line_position = (int(points_labels[label_keys[j]][0, frame_counter]), int(points_labels[label_keys[j]][1, frame_counter]))
                    cv2.line(small_image, mouse_click_position, line_position, rectangle_color, thickness=3)

    cv2.imshow(Image_name, small_image)
    return


def circle_positioning(event, x, y, flags, param):
    global points_labels, current_click, frame_counter

    if event == cv2.EVENT_LBUTTONDOWN:
        points_labels[label_keys[current_click]][:, frame_counter] = np.array([x, y])
        draw_points_and_lines()
    return


def left_front_corner_choice(*args):
    global current_click
    current_click = 0
    active_points[frame_counter, 0] = True
    active_points[frame_counter, 4] = False
    print('current_click : ', current_click)
    return

def right_front_corner_choice(*args):
    global current_click
    current_click = 1
    active_points[frame_counter, 1] = True
    active_points[frame_counter, 5] = False
    print('current_click : ', current_click)
    return

def right_back_corner_choice(*args):
    global current_click
    current_click = 2
    active_points[frame_counter, 2] = True
    active_points[frame_counter, 6] = False
    print('current_click : ', current_click)
    return

def left_back_corner_choice(*args):
    global current_click
    current_click = 3
    active_points[frame_counter, 3] = True
    active_points[frame_counter, 7] = False
    print('current_click : ', current_click)
    return

def left_front_border_choice(*args):
    global current_click
    current_click = 4
    active_points[frame_counter, 4] = True
    active_points[frame_counter, 0] = False
    print('current_click : ', current_click)
    return

def right_front_border_choice(*args):
    global current_click
    current_click = 5
    active_points[frame_counter, 5] = True
    active_points[frame_counter, 1] = False
    print('current_click : ', current_click)
    return

def right_back_border_choice(*args):
    global current_click
    current_click = 6
    active_points[frame_counter, 6] = True
    active_points[frame_counter, 2] = False
    print('current_click : ', current_click)
    return

def left_back_border_choice(*args):
    global current_click
    current_click = 7
    active_points[frame_counter, 7] = True
    active_points[frame_counter, 3] = False
    print('current_click : ', current_click)
    return


############################### code beginning #######################################################################
global small_image, image, image_bidon

circle_radius = 5
rectangle_color = (1, 1, 1)
circle_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (100, 0, 0), (0, 100, 0), (0, 0, 100), (100, 100, 0)]
Image_name = "Video"
Trackbar_name = "Frames"
ratio_image = 1.5

movie_file = 'PI world v1 ps1.mp4'
frames, num_frames = load_video_frames(movie_file)
frames_clone = frames.copy()
points_labels = {"left_front_corner": np.zeros((2, len(frames))),
                 "right_front_corner": np.zeros((2, len(frames))),
                 "right_back_corner": np.zeros((2, len(frames))),
                 "left_back_corner": np.zeros((2, len(frames))),
                 "left_front_border": np.zeros((2, len(frames))),
                 "right_front_border": np.zeros((2, len(frames))),
                 "right_back_border": np.zeros((2, len(frames))),
                 "left_back_border": np.zeros((2, len(frames)))}
label_keys = [key for key in points_labels.keys()]
current_click = 0
active_points = np.zeros((num_frames, 8))
neighbors = [[1, 5, 3, 7],
             [0, 4, 2, 6],
             [1, 5, 3, 7],
             [0, 4, 2, 6],
             [1, 5, 3, 7],
             [0, 4, 2, 6],
             [1, 5, 3, 7],
             [0, 4, 2, 6],]

def nothing(x):
    return

cv2.namedWindow(Image_name)
cv2.createTrackbar('Frames', Image_name, 0, num_frames, nothing)
cv2.createButton("Left Front Corner (LFC)", left_front_corner_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Right Front Corner (RFC)", right_front_corner_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Right Back Corner (RBC)", right_back_corner_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Left Back Corner (LBC)", left_back_corner_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Left Front Border (LFB)", left_front_border_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Right Front Border (RFB)", right_front_border_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Right Back Border (RBB)", right_back_border_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Left Back Border (LBB)", left_back_border_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.setMouseCallback(Image_name, circle_positioning)

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

with open(f'output/{movie_file[:-4]}_labeling_points.pkl', 'wb') as handle:
    pickle.dump([points_labels, active_points], handle)















