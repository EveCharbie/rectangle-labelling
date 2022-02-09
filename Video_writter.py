
import cv2
import numpy as np
from tqdm.notebook import tqdm
from skvideo.io import FFmpegWriter
import pickle
import subprocess as sp


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


def write_labeled_video(output_file, frames_clone, fps):
    video = FFmpegWriter(output_file, inputdict={'-r': str(fps)}, outputdict={'-r': str(fps)})

    frames_small = np.array(frames_clone)

    for frame_num in tqdm(np.arange(frames_small.shape[0])):
        video.writeFrame(frames_small[frame_num, :, :])

    video.close()


def draw_points_and_lines():
    global points_labels, circle_colors, circle_radius, frame_counter, small_image, active_points, rectangle_color

    for i in range(8):
        if active_points[frame_counter, i] == True:
            mouse_click_position = (int(points_labels[label_keys[i]][0, frame_counter]),
                                    int(points_labels[label_keys[i]][1, frame_counter]))
            cv2.circle(small_image, mouse_click_position, circle_radius, color=circle_colors[i], thickness=-1)
            for j in neighbors[i]:
                if active_points[frame_counter, j] == True:
                    line_position = (int(points_labels[label_keys[j]][0, frame_counter]), int(points_labels[label_keys[j]][1, frame_counter]))
                    cv2.line(small_image, mouse_click_position, line_position, rectangle_color, thickness=3)

    return

############################### code beginning #######################################################################
global frame_counter

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
[points_labels, active_points] = pickle.load(open(f'output/{movie_file[:-4]}_labeling_points.pkl', "rb"))
label_keys = [key for key in points_labels.keys()]
neighbors = [[1, 5, 3, 7],
             [0, 4, 2, 6],
             [1, 5, 3, 7],
             [0, 4, 2, 6],
             [1, 5, 3, 7],
             [0, 4, 2, 6],
             [1, 5, 3, 7],
             [0, 4, 2, 6],]

width, height, rgb = np.shape(frames[0])

output_file = f'output/{movie_file[:-4]}_labeled.mp4'
fps = 30
# video_writer = cv2.VideoWriter_fourcc(*'XVID') # ('M', 'P', 'E', 'G') # ('m', 'p', '4', 'v')  # ('M', 'P', '4', 'V')
# out_video = cv2.VideoWriter(output_file, video_writer, fps, (width,height))

for frame_counter in range(num_frames):

    small_image = cv2.resize(frames[frame_counter], (int(round(width / ratio_image)), int(round(height / ratio_image))))
    draw_points_and_lines()
    frames_clone[frame_counter] = small_image #  frame_gbr
    # out_video.write(small_image)

# out_video.release()

write_labeled_video(output_file, frames_clone, fps)



# #
# command = [ "ffmpeg", # on Linux ans Mac OS, # FFMPEG_BIN = "ffmpeg.exe" # on Windows
#         '-y', # (optional) overwrite output file if it exists
#         '-f', 'rawvideo',
#         '-vcodec','rawvideo',
#         '-s', '420x360', # size of one frame
#         '-pix_fmt', 'rgb24',
#         '-r', f'{fps}', # frames per second
#         '-i', '-', # The imput comes from a pipe
#         '-an', # Tells FFMPEG not to expect any audio
#         '-vcodec', 'mpeg'",
#         'my_output_videofile.mp4' ]
#
# pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
















