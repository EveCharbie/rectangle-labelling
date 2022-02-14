
import cv2
import numpy as np
from tqdm.notebook import tqdm
import pickle


def draw_points_and_lines():
    global points_labels, circle_colors, circle_radius, iFrame, image, clone, active_points, rectangle_color

    clone = image.copy()

    for i in range(len(active_points)):
        if active_points[i] == True:
            mouse_click_position = (int(points_labels[label_keys[i]][0, iFrame]), int(points_labels[label_keys[i]][1, iFrame]))
            cv2.circle(clone, mouse_click_position, circle_radius, color=circle_colors[i], thickness=-1) # clone
            for j in neighbors[i]:
                if active_points[j] == True:
                    line_position = (int(points_labels[label_keys[j]][0, iFrame]), int(points_labels[label_keys[j]][1, iFrame]))
                    cv2.line(clone, mouse_click_position, line_position, rectangle_color, thickness=1) # clone

    return


def circle_positioning(event, x, y, flags, param):
    global points_labels, current_click, iFrame

    if event == cv2.EVENT_LBUTTONDOWN:
        points_labels[label_keys[current_click]][:, iFrame] = np.array([x, y])
        draw_points_and_lines()
    return


def left_front_corner_choice(*args):
    global current_click
    current_click = 0
    active_points[0] = True
    active_points[4] = False
    print('current_click : ', current_click)
    return

def right_front_corner_choice(*args):
    global current_click
    current_click = 1
    active_points[1] = True
    active_points[5] = False
    print('current_click : ', current_click)
    return

def right_back_corner_choice(*args):
    global current_click
    current_click = 2
    active_points[2] = True
    active_points[6] = False
    print('current_click : ', current_click)
    return

def left_back_corner_choice(*args):
    global current_click
    current_click = 3
    active_points[3] = True
    active_points[7] = False
    print('current_click : ', current_click)
    return

def left_front_border_choice(*args):
    global current_click
    current_click = 4
    active_points[4] = True
    active_points[0] = False
    print('current_click : ', current_click)
    return

def right_front_border_choice(*args):
    global current_click
    current_click = 5
    active_points[5] = True
    active_points[1] = False
    print('current_click : ', current_click)
    return

def right_back_border_choice(*args):
    global current_click
    current_click = 6
    active_points[6] = True
    active_points[2] = False
    print('current_click : ', current_click)
    return

def left_back_border_choice(*args):
    global current_click
    current_click = 7
    active_points[7] = True
    active_points[3] = False
    print('current_click : ', current_click)
    return



frames = np.array([0])
points_labels = {"left_front_corner": np.zeros((2, len(frames))),
                 "right_front_corner": np.zeros((2, len(frames))),
                 "right_back_corner": np.zeros((2, len(frames))),
                 "left_back_corner": np.zeros((2, len(frames))),
                 "left_front_border": np.zeros((2, len(frames))),
                 "right_front_border": np.zeros((2, len(frames))),
                 "right_back_border": np.zeros((2, len(frames))),
                 "left_back_border": np.zeros((2, len(frames)))}
label_keys = [key for key in points_labels.keys()]
current_click = None
active_points = [False for i in range(8)]
iFrame = 0
neighbors = [[1, 5, 3, 7],
             [0, 4, 2, 6],
             [1, 5, 3, 7],
             [0, 4, 2, 6],
             [1, 5, 3, 7],
             [0, 4, 2, 6],
             [1, 5, 3, 7],
             [0, 4, 2, 6],]



circle_radius = 5
rectangle_color = (100, 100, 100)
circle_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), rectangle_color, rectangle_color, rectangle_color, rectangle_color]
Image_name = "Image_bidon.png"


############################### code beginning #######################################################################
global clone

# image = cv2.imread('../input/PI world v1 ps1_181.jpg')
file = open(f"../output/PI world v1 ps1_181_undistorted_images.pkl", "rb")
image = pickle.load(file)

clone = image.copy()
cv2.namedWindow(Image_name)
cv2.createButton("Left Front Corner (LFC)", left_front_corner_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Right Front Corner (RFC)", right_front_corner_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Right Back Corner (RBC)", right_back_corner_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Left Back Corner (LBC)", left_back_corner_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Left Front Border (LFB)", left_front_border_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Right Front Border (RFB)", right_front_border_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Right Back Border (RBB)", right_back_border_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.createButton("Left Back Border (LBB)", left_back_border_choice, None, cv2.QT_PUSH_BUTTON, 0)
cv2.setMouseCallback(Image_name, circle_positioning)

while True:
    ratio_image = 1.5
    width, height, rgb = np.shape(clone)
    small_image = cv2.resize(clone, (int(round(width / ratio_image)), int(round(height / ratio_image))))
    cv2.imshow(Image_name, small_image)
    key = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()
