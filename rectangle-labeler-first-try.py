
import cv2
import numpy as np
from tqdm.notebook import tqdm


def load_video_frames(video_file, num_frames=None):
    video = cv2.VideoCapture(video_file)
    frames = []

    if num_frames is None:
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(num_frames), desc='Loading video'):
        ret, frame = video.read()

        frames.append(frame)
        # key = cv2.waitKey(1)
    video.release()

    return frames


def set_frame_counter(frame_counter, num_frames):
    # if the user has scrolled past the end, go to the beginning
    if frame_counter == num_frames:
        frame_counter = 0
        # if user has scrolled to the left of the beginning, go to the end
    elif frame_counter == -1:
        frame_counter = num_frames - 1

    return frame_counter


def write_labeled_video(output_file, frames_clone, fps):
    video = FFmpegWriter(output_file,
                         inputdict={'-r': str(fps)}, outputdict={'-r': str(fps)})

    frames = np.array(frames_clone)

    for frame_num in tqdm(np.arange(frames.shape[0])):
        video.writeFrame(frames[frame_num, :, :])

    video.close()


def draw_points_and_lines():
    global points_labels, circle_colors, circle_radius, frame_counter, image, image_clone, active_points, rectangle_color, frames_clone, frame_counter

    image_clone = image.copy()
    frames_clone[frame_counter] = image_clone

    for i in range(len(active_points)):
        if active_points[i] == True:
            mouse_click_position = (int(points_labels[label_keys[i]][0, frame_counter]), int(points_labels[label_keys[i]][1, frame_counter]))
            cv2.circle(image_clone, mouse_click_position, circle_radius, color=circle_colors[i], thickness=-1) # image_clone
            for j in neighbors[i]:
                if active_points[j] == True:
                    line_position = (int(points_labels[label_keys[j]][0, frame_counter]), int(points_labels[label_keys[j]][1, frame_counter]))
                    cv2.line(image_clone, mouse_click_position, line_position, rectangle_color, thickness=1) # image_clone

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


############################### code beginning #######################################################################
global image_clone

circle_radius = 5
rectangle_color = (100, 100, 100)
circle_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), rectangle_color, rectangle_color, rectangle_color, rectangle_color]
Image_name = "frame"

movie_file = 'PI world v1 ps1.mp4'
# image = cv2.imread("Image_bidon.png")
frames = load_video_frames(movie_file)
frames_clone = frames.copy()
num_frames = len(frames)
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
neighbors = [[1, 5, 3, 7],
             [0, 4, 2, 6],
             [1, 5, 3, 7],
             [0, 4, 2, 6],
             [1, 5, 3, 7],
             [0, 4, 2, 6],
             [1, 5, 3, 7],
             [0, 4, 2, 6],]

cv2.namedWindow(Image_name)
cv2.createTrackbar(Image_name, 'Video', 0, num_frames, on_trackbar)
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
while playVideo == True:

    frame_counter = cv2.getTrackbarPos(Image_name, 'Video')
    frames_clone[frame_counter] = frames[frame_counter].copy()
    image_clone = frames_clone[frame_counter]

    cv2.imshow(Image_name, image_clone)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(','):  # if `<` then go back
        frame_counter -= 1
        frame_counter = set_frame_counter(frame_counter, num_frames)
        cv2.setTrackbarPos(Image_name, "Video", frame_counter)

    elif key == ord('.'):  # if `>` then advance
        frame_counter += 1
        frame_counter = set_frame_counter(frame_counter, num_frames)
        cv2.setTrackbarPos(Image_name, "Video", frame_counter)

    elif key == ord('x'):  # if `x` then quit
        playVideo = False

cv2.destroyAllWindows()




output_file = f'{movie_file[:-4]}_labeled.mp4'
fps = 30
write_labeled_video(output_file, frames_clone, fps)



#
# def loadTiffBatch(video_dir, start, size):
#     bordersize = 50
#
#     batch = []
#
#     for i in range(start, start + size):
#         filename = os.path.join(video_dir, 'frame' + str(i) + '.tiff')
#         img = cv2.imread(filename)
#         border = cv2.copyMakeBorder(
#             img,
#             top=bordersize,
#             bottom=bordersize,
#             left=bordersize,
#             right=bordersize,
#             borderType=cv2.BORDER_CONSTANT,
#             value=[255, 255, 255]
#         )
#         batch.append(border)
#
#     return batch













