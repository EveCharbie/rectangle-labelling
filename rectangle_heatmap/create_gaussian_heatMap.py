
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import cv2


def points_to_gaussian_heatmap(centers, height, width, scale):
    # from : https://stackoverflow.com/questions/44945111/how-to-efficiently-compute-the-heat-map-of-two-gaussian-distribution-in-python

    gaussians = []
    for y, x in centers:
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
image_heighgt = 428
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

gaze_position_labels = f'../output/{movie_name[:-4]}_labeling_points.pkl'
out_path = '/home/user/Documents/Programmation/Eve/rectangle-labelling/rectangle-labelling/output/Results/'
subject_name = 'GuSe'
move_names = ['4-o', '8--1o', '8--o', '8--1o']

file = open(gaze_position_labels, "rb")
points_labels, active_points, curent_AOI_label, csv_eye_tracking = pickle.load(file)

zeros_clusters_index = curent_AOI_label["Not an acrobatics"][:-1] - curent_AOI_label["Not an acrobatics"][1:]
zeros_clusters_index = np.hstack((0, zeros_clusters_index))
start_of_move_index = np.where(zeros_clusters_index == -1)
end_of_move_index = np.where(zeros_clusters_index == 1)

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
    gaze_total_move = start - end
    number_of_trampoline_bed = 0
    number_of_wall_front = 0
    number_of_wall_back = 0
    number_of_ceiling = 0
    for j in range(start, end):
        if curent_AOI_label["Trampoline bed"] == 1:
            centers_gaze_bed_i.append((csv_eye_tracking[j, 5], csv_eye_tracking[j, 6]))
            number_of_trampoline_bed += 1
        elif curent_AOI_label["Wall front"][j] == 1:
            number_of_wall_front += 1
        elif curent_AOI_label["Wall back"][j] == 1:
            number_of_wall_back += 1
        elif curent_AOI_label["Ceiling"][j] == 1:
            number_of_ceiling += 1
    centers_gaze_bed[i] = centers_gaze_bed_i
    img = points_to_gaussian_heatmap(centers_gaze_bed, image_width, image_heighgt, gaussian_width)
    plt.imshow(img)
    plt.show()

    move_summary[i] = {"centers": centers_gaze_bed_i,
                       "heat map": img,
                       "trampoline bed proportions": number_of_trampoline_bed/gaze_total_move,
                       "wall front proportions": number_of_wall_front/gaze_total_move,
                       "wall back proportions": number_of_wall_back/gaze_total_move,
                       "ceiling proportions": number_of_ceiling/gaze_total_move}

    with open(f'{out_path}/{subject_name}/{gaze_position_labels[:-20]}_heat_map_{i}.pkl', 'wb') as handle:
        pickle.dump(move_summary[i], handle)


with open(f'../output/{gaze_position_labels[:-20]}_heat_map.pkl', 'wb') as handle:
    pickle.dump(move_summary, handle)


# img = points_to_gaussian_heatmap(centers_gaze_bed, image_width, image_heighgt, gaussian_width)
#
# plt.imshow(img)
# plt.show()
