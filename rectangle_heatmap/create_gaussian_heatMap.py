
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import cv2
import pickle
import os
import pandas as pd


def load_pupil(gaze_position_labels, eye_tracking_data_path):

    file = open(gaze_position_labels, "rb")
    points_labels, active_points, curent_AOI_label, csv_eye_tracking = pickle.load(file)

    filename_gaze = eye_tracking_data_path + 'gaze.csv'
    filename_timestamps = eye_tracking_data_path + 'world_timestamps.csv'
    filename_info = eye_tracking_data_path + 'info.json'


    csv_gaze_read = np.char.split(pd.read_csv(filename_gaze, sep='\t').values.astype('str'), sep=',')
    timestamp_image_read = np.char.split(pd.read_csv(filename_timestamps, sep='\t').values.astype('str'), sep=',')
    timestamp_image = np.zeros((len(timestamp_image_read, )))
    for i in range(len(timestamp_image_read)):
        timestamp_image[i] = timestamp_image_read[i][0][2]
    info = np.char.split(pd.read_csv(filename_info, sep='\t').values.astype('str'), sep=',')
    serial_number_str = info[15][0][0]
    num_quote = 0
    for pos, char in enumerate(serial_number_str):
        if char == '"':
            num_quote += 1
            if num_quote == 3:
                SCENE_CAMERA_SERIAL_NUMBER = serial_number_str[pos+1:pos+6]
                break

    csv_eye_tracking = np.zeros((len(csv_gaze_read), 7))
    for i in range(len(csv_gaze_read)):
        csv_eye_tracking[i, 0] = float(csv_gaze_read[i][0][2])  # timestemp
        csv_eye_tracking[i, 1] = int(round(float(csv_gaze_read[i][0][3])))  # pos_x
        csv_eye_tracking[i, 2] = int(round(float(csv_gaze_read[i][0][4])))  # pos_y
        csv_eye_tracking[i, 4] = np.argmin(np.abs(csv_eye_tracking[i, 0] - timestamp_image))  # closest image timestemp

    # embed()
    # plt.figure()
    # plt.plot(csv_eye_tracking[:, 0], '-b')
    # plt.plot(timestamp_image, '-r')
    # plt.show()


    time_stamps_eye_tracking = np.zeros((len(timestamp_image),))
    time_stamps_eye_tracking_index_on_pupil = np.zeros((len(timestamp_image),))
    for i in range(len(timestamp_image)):
        time_stamps_eye_tracking_index_on_pupil[i] = np.argmin(np.abs(csv_eye_tracking[:, 0] - float(timestamp_image[i])))


    zeros_clusters_index = curent_AOI_label["Not an acrobatics"][:-1] - curent_AOI_label["Not an acrobatics"][1:]
    zeros_clusters_index = np.hstack((0, zeros_clusters_index))

    end_of_cluster_index_image = np.where(zeros_clusters_index == -1)[0].tolist()
    start_of_cluster_index_image = np.where(zeros_clusters_index == 1)[0].tolist()

    start_of_move_index_image = []
    end_of_move_index_image = []
    start_of_jump_index_image = []
    end_of_jump_index_image = []
    for i in range(len(start_of_cluster_index_image)):
        if curent_AOI_label["Jump"][start_of_cluster_index_image[i] + 1] == 1:
            start_of_jump_index_image += [start_of_cluster_index_image[i]]
            end_of_jump_index_image += [end_of_cluster_index_image[i]]
        else:
            start_of_move_index_image += [start_of_cluster_index_image[i]]
            end_of_move_index_image += [end_of_cluster_index_image[i]]


    end_of_move_index = time_stamps_eye_tracking_index_on_pupil[end_of_move_index_image]
    start_of_move_index = time_stamps_eye_tracking_index_on_pupil[start_of_move_index_image]
    end_of_jump_index = time_stamps_eye_tracking_index_on_pupil[end_of_jump_index_image]
    start_of_jump_index = time_stamps_eye_tracking_index_on_pupil[start_of_jump_index_image]

    return curent_AOI_label, csv_eye_tracking, csv_eye_tracking, start_of_move_index, end_of_move_index, start_of_jump_index, end_of_jump_index, start_of_move_index_image, end_of_move_index_image, start_of_jump_index_image, end_of_jump_index_image


def points_to_gaussian_heatmap(centers, height, width, scale):
    # from : https://stackoverflow.com/questions/44945111/how-to-efficiently-compute-the-heat-map-of-two-gaussian-distribution-in-python

    gaussians = []
    for x, y in centers:
        s = np.eye(2) * scale
        g = multivariate_normal(mean=(x, y), cov=s)
        gaussians.append(g)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.stack([xx.ravel(), yy.ravel()]).T

    # evaluate kernels at grid points
    zz = sum(g.pdf(xxyy) for g in gaussians)

    img = zz.reshape((height, width))
    return img

def put_lines_on_fig():
    plt.plot(np.array([0, 214]), np.array([0, 0]), '-w', linewidth=1)
    plt.plot(np.array([214, 214]), np.array([214, 428]), '-w', linewidth=1)
    plt.plot(np.array([0, 214]), np.array([428, 428]), '-w', linewidth=1)
    plt.plot(np.array([0, 0]), np.array([0, 428]), '-w', linewidth=1)

    plt.plot(np.array([53, 53]), np.array([0, 428]), '-w', linewidth=1)
    plt.plot(np.array([161, 161]), np.array([0, 428]), '-w', linewidth=1)
    plt.plot(np.array([0, 214]), np.array([107, 107]), '-w', linewidth=1)
    plt.plot(np.array([0, 214]), np.array([322, 322]), '-w', linewidth=1)
    plt.plot(np.array([53, 161]), np.array([160, 160]), '-w', linewidth=1)
    plt.plot(np.array([53, 161]), np.array([268, 268]), '-w', linewidth=1)
    plt.plot(np.array([107 - 25, 107 + 25]), np.array([214, 214]), '-w', linewidth=1)
    plt.plot(np.array([107, 107]), np.array([214 - 25, 214 + 25]), '-w', linewidth=1)
    return

def run_create_heatmaps(move_names, start_of_move_index_image, end_of_move_index_image, curent_AOI_label, csv_eye_tracking):
    image_width = 214
    image_height = 428
    gaussian_width = 50



    if len(move_names) != len(start_of_move_index_image) or len(move_names) != len(end_of_move_index_image):
        plt.figure()
        plt.plot(curent_AOI_label["Not an acrobatics"], '-k')
        plt.plot(curent_AOI_label["Jump"], '-r')
        plt.show()
        raise RuntimeError("Not the right number of skills!")

    move_summary = [{} for i in range(len(move_names))]

    centers_gaze_bed = [[] for i in range(len(move_names))]
    gaze_wall_front_index = [[] for i in range(len(move_names))]
    gaze_wall_back_index = [[] for i in range(len(move_names))]
    gaze_ceiling_index = [[] for i in range(len(move_names))]
    for i in range(len(move_names)):
        start = start_of_move_index_image[i]
        end = end_of_move_index_image[i]
        centers_gaze_bed_i = []
        gaze_total_move = end - start
        number_of_trampoline_bed = 0
        number_of_wall_front = 0
        number_of_wall_back = 0
        number_of_ceiling = 0
        number_of_trampoline = 0
        for j in range(start, end):
            if curent_AOI_label["Trampoline bed"][j] == 1:
                index_gaze = np.where(csv_eye_tracking[:, 4] == j)[0]
                for k in index_gaze:
                    if csv_eye_tracking[k, 5] != 0 and csv_eye_tracking[k, 6] != 0:
                        if move_orientation[i] < 0:
                            centers_gaze_bed_i.append((image_width - csv_eye_tracking[k, 5], image_height - csv_eye_tracking[k, 6]))
                        else:
                            centers_gaze_bed_i.append((csv_eye_tracking[k, 5], csv_eye_tracking[k, 6]))
                number_of_trampoline_bed += 1
            elif curent_AOI_label["Wall front"][j] == 1:
                if move_orientation[i] < 0:
                    number_of_wall_back += 1
                else:
                    number_of_wall_front += 1
            elif curent_AOI_label["Wall back"][j] == 1:
                if move_orientation[i] < 0:
                    number_of_wall_front += 1
                else:
                    number_of_wall_back += 1
            elif curent_AOI_label["Ceiling"][j] == 1:
                number_of_ceiling += 1
            elif curent_AOI_label["Trampoline"][j] == 1:
                number_of_trampoline += 1

        centers_gaze_bed[i] = centers_gaze_bed_i

        plt.figure()
        put_lines_on_fig()
        img = points_to_gaussian_heatmap(centers_gaze_bed[i], image_height, image_width, gaussian_width)
        plt.imshow(img, cmap=plt.get_cmap('plasma'))
        plt.title(f"{subject_name}({subject_expertise}): {move_names[i]}")
        plt.axis('off')

        move_summary[i] = {"movement_name": move_names[i],
                           "subject_name": subject_name,
                           "movie_name": movie_name,
                           "centers": centers_gaze_bed_i,
                           "heat_map": img,
                           "trampoline_bed_proportions": number_of_trampoline_bed/gaze_total_move,
                           "wall_front_proportions": number_of_wall_front/gaze_total_move,
                           "wall_back_proportions": number_of_wall_back/gaze_total_move,
                           "ceiling_proportions": number_of_ceiling/gaze_total_move}

        if not os.path.exists(f'{out_path}/{subject_name}'):
            os.makedirs(f'{out_path}/{subject_name}')
        if not os.path.exists(f'{out_path}/{subject_name}/{move_names[i]}'):
            os.makedirs(f'{out_path}/{subject_name}/{move_names[i]}')

        with open(f'{out_path}/{subject_name}/{move_names[i]}/{movie_name}_heat_map_{repetition_number[i]}.pkl', 'wb') as handle:
            pickle.dump(move_summary[i], handle)

        plt.savefig(f"{out_path}/{subject_name}/{move_names[i]}/{movie_name}_heat_map_{repetition_number[i]}.png", format="png")
        # plt.show()
        print(f"Generated {subject_name}({subject_expertise}): {move_names[i]}")

    with open(f'{gaze_position_labels[:-20]}_heat_map.pkl', 'wb') as handle:
        pickle.dump(move_summary, handle)


def __main__():

    if os.path.exists('/home/user'):
        root_path = '/home/user'
    elif os.path.exists('/home/fbailly'):
        root_path = '/home/fbailly'
    elif os.path.exists('/home/charbie'):
        root_path = '/home/charbie'

    csv_name = root_path + "/Documents/Programmation/rectangle-labelling/Trials_name_mapping.csv"
    csv_table = np.char.split(pd.read_csv(csv_name, sep='\t').values.astype('str'), sep=',')

    for i_trial in range(len(csv_table)):
        if csv_table[i_trial][0][12] != 'True':
            continue
        movie_path = "/home/user/disk/Eye-tracking/PupilData/points_labeled/"
        movie_name = csv_table[i_trial][0][7].replace('.', '_')
        gaze_position_labels = movie_path + movie_name + "_labeling_points.pkl"
        out_path = '/home/user/disk/Eye-tracking/Results'
        subject_name = csv_table[i_trial][0][0]
        move_names = csv_table[i_trial][0][1].split(" ")
        repetition_number = csv_table[i_trial][0][2].split(" ")
        move_orientation = [int(x) for x in csv_table[i_trial][0][3].split(" ")]
        subject_expertise = csv_table[i_trial][0][9]

        curent_AOI_label, csv_eye_tracking, csv_eye_tracking, start_of_move_index, end_of_move_index, start_of_jump_index, end_of_jump_index, start_of_move_index_image, end_of_move_index_image, start_of_jump_index_image, end_of_jump_index_image = load_pupil(
            gaze_position_labels, eye_tracking_data_path)


        run_create_heatmaps(move_names, start_of_move_index_image, end_of_move_index_image, curent_AOI_label, csv_eye_tracking)
    