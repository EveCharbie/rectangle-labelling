
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import cv2
import pickle
import os


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



image_width = 214
image_height = 428
gaussian_width = 50

plt.figure()
plt.plot(np.array([0, 214]), np.array([0, 0]), '-k', linewidth=1)
plt.plot(np.array([214, 214]), np.array([214, 428]), '-k', linewidth=1)
plt.plot(np.array([0, 214]), np.array([428, 428]), '-k', linewidth=1)
plt.plot(np.array([0, 0]), np.array([0, 428]), '-k', linewidth=1)

plt.plot(np.array([53, 53]), np.array([0, 428]), '-k', linewidth=1)
plt.plot(np.array([161, 161]), np.array([0, 428]), '-k', linewidth=1)
plt.plot(np.array([0, 214]), np.array([107, 107]), '-k', linewidth=1)
plt.plot(np.array([0, 214]), np.array([322, 322]), '-k', linewidth=1)
plt.plot(np.array([53, 161]), np.array([160, 160]), '-k', linewidth=1)
plt.plot(np.array([53, 161]), np.array([268, 268]), '-k', linewidth=1)
plt.plot(np.array([107-25, 107+25]), np.array([214, 214]), '-k', linewidth=1)
plt.plot(np.array([107, 107]), np.array([214-25, 214+25]), '-k', linewidth=1)

movie_path = "../output/"
#########################################################################################
# "df297219_0_0-43_588" GuSe
# "d7b37ec2_0_0-37_269" AnBe
movie_name = "d7b37ec2_0_0-37_269"
gaze_position_labels = movie_path + movie_name + "_labeling_points.pkl"
out_path = '/home/user/Documents/Programmation/rectangle-labelling/output/Results'
subject_name = 'AnBe' # 'GuSe'
move_names = ['4-o'] # , '8--1o', '8--o', '8--1o']
move_orientation = [-1] # AnBe # [+1] # GuSe

file = open(gaze_position_labels, "rb")
points_labels, active_points, curent_AOI_label, csv_eye_tracking = pickle.load(file)

zeros_clusters_index = curent_AOI_label["Not an acrobatics"][:-1] - curent_AOI_label["Not an acrobatics"][1:]
zeros_clusters_index = np.hstack((0, zeros_clusters_index))
end_of_move_index = np.where(zeros_clusters_index == -1)[0].tolist() # [np.where(curent_AOI_label["Wall front"] == 1)[0][0]] ###############3
start_of_move_index = np.where(zeros_clusters_index == 1)[0].tolist() # [np.where(points_labels["0"][0, :] != 0)[0][-1]] ##############

if len(move_names) != len(start_of_move_index) or len(move_names) != len(end_of_move_index):
    raise RuntimeError("Not the right number of skills!")

move_summary = [{} for i in range(len(move_names))]

centers_gaze_bed = [[] for i in range(len(move_names))]
gaze_wall_front_index = [[] for i in range(len(move_names))]
gaze_wall_back_index = [[] for i in range(len(move_names))]
gaze_ceiling_index = [[] for i in range(len(move_names))]
for i in range(len(move_names)):
    start = start_of_move_index[i]
    end = end_of_move_index[i]
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
    img = points_to_gaussian_heatmap(centers_gaze_bed[i], image_height, image_width, gaussian_width)
    plt.imshow(img)
    plt.show()

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

    with open(f'{out_path}/{subject_name}/{move_names[i]}/{movie_name}_heat_map_{i}.pkl', 'wb') as handle:
        pickle.dump(move_summary[i], handle)


with open(f'{gaze_position_labels[:-20]}_heat_map.pkl', 'wb') as handle:
    pickle.dump(move_summary, handle)


# img = points_to_gaussian_heatmap(centers_gaze_bed, image_width, image_heighgt, gaussian_width)
#
# plt.imshow(img)
# plt.show()
