
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from create_gaussian_heatMap import put_lines_on_fig
from scipy.stats import mannwhitneyu


def plot_img_heatmap(fig_name, img):
    plt.figure()
    put_lines_on_fig()
    plt.imshow(img, cmap=plt.get_cmap('plasma'))
    plt.axis('off')
    plt.savefig(fig_name, dpi=300, format="png")
    plt.show()
    return


if os.path.exists('/home/user'):
    root_path = '/home/user'
elif os.path.exists('/home/fbailly'):
    root_path = '/home/fbailly'
elif os.path.exists('/home/charbie'):
    root_path = '/home/charbie'

subjects = ['GuSe', 'AlLe']
moves = ['4-', '4-o']
repetitions = [5, 5]

heatmap_means = [np.zeros((428, 214)) for i in range(len(subjects))]
gaze_AOI = {'trampoline_bed_proportions': [np.zeros((repetitions[i], )) for i in range(len(subjects))],
            'wall_proportions': [np.zeros((repetitions[i], )) for i in range(len(subjects))],
            'ceiling_proportions': [np.zeros((repetitions[i], )) for i in range(len(subjects))]}
gaze_AOI_mean_trampo = [0, 0]
gaze_AOI_std_trampo = [0, 0]
gaze_AOI_mean_walls = [0, 0]
gaze_AOI_std_walls = [0, 0]
gaze_AOI_mean_ceiling = [0, 0]
gaze_AOI_std_ceiling = [0, 0]
for isubject in range(len(subjects)):
    path = root_path + f"/Documents/Programmation/rectangle-labelling/Results_tempo/{subjects[isubject]}/{moves[isubject]}"
    files_candidates = os.listdir(path)
    files=[]
    for i in range(len(files_candidates)):
        if files_candidates[i].endswith('.pkl'):
            files += [files_candidates[i]]

    for imove in range(len(files)):
        file = open(path + '/' + files[imove], "rb")
        data = pickle.load(file)
        heatmap_means[isubject] += data['heat_map']

        gaze_AOI['trampoline_bed_proportions'][isubject][imove] = data['trampoline_bed_proportions'] * 100
        gaze_AOI['wall_proportions'][isubject][imove] = data['wall_front_proportions'] * 100 + data['wall_back_proportions'] * 100
        gaze_AOI['ceiling_proportions'][isubject][imove] = data['ceiling_proportions'] * 100

        plot_img_heatmap(path + '/' + files[imove][:-4] + '.png', data['heat_map'])

    heatmap_means[isubject] /= len(files)
    gaze_AOI_mean_trampo[isubject] = np.mean(gaze_AOI['trampoline_bed_proportions'][isubject])
    gaze_AOI_std_trampo[isubject] = np.std(gaze_AOI['trampoline_bed_proportions'][isubject])
    gaze_AOI_mean_walls[isubject] = np.mean(gaze_AOI['wall_proportions'][isubject])
    gaze_AOI_std_walls[isubject] = np.std(gaze_AOI['wall_proportions'][isubject])
    gaze_AOI_mean_ceiling[isubject] = np.mean(gaze_AOI['ceiling_proportions'][isubject])
    gaze_AOI_std_ceiling[isubject] = np.std(gaze_AOI['ceiling_proportions'][isubject])

plt.figure()
put_lines_on_fig()
img = heatmap_means[0] - heatmap_means[1]
plt.imshow(img, cmap=plt.get_cmap('plasma'))
plt.title("Différence")
plt.axis('off')
plt.colorbar()
plt.savefig(root_path + "/Documents/Programmation/rectangle-labelling/rectangle_heatmap/Results_tempo/Difference_GuSe_AlLe.png", dpi=300, format="png")
plt.show()


U_trampoline, p_trampoline = mannwhitneyu(gaze_AOI['trampoline_bed_proportions'][0], gaze_AOI['trampoline_bed_proportions'][1])
U_walls, p_walls = mannwhitneyu(gaze_AOI['wall_proportions'][0], gaze_AOI['wall_proportions'][1])
U_ceiling, p_ceiling = mannwhitneyu(gaze_AOI['ceiling_proportions'][0], gaze_AOI['ceiling_proportions'][1])

print(p_trampoline*3)
print(p_walls*3)
print(p_ceiling*3)

color_1 = "#6BC2D3FF" # "#FFDB8DFF"
color_2 = "#031859FF" # "#6BB62FF"
plt.figure(figsize=(4, 3))
plt.bar(np.array([0.8, 1.8, 2.8]), np.array([gaze_AOI_mean_trampo[0], gaze_AOI_mean_walls[0], gaze_AOI_mean_ceiling[0]]),
        yerr=np.array([gaze_AOI_std_trampo[0], gaze_AOI_std_walls[0], gaze_AOI_std_ceiling[0]]), alpha=0.7,
        color=color_1, ecolor=color_1, width=0.4)
plt.bar(np.array([1.2, 2.2, 3.2]), np.array([gaze_AOI_mean_trampo[1], gaze_AOI_mean_walls[1], gaze_AOI_mean_ceiling[1]]),
        yerr=np.array([gaze_AOI_std_trampo[1], gaze_AOI_std_walls[1], gaze_AOI_std_ceiling[1]]), alpha=0.7,
        color=color_2, ecolor=color_2, width=0.4)

plt.plot(np.ones((5,))*0.8, gaze_AOI['trampoline_bed_proportions'][0], '.', color=color_1)
plt.plot(np.ones((5,))*1.8, gaze_AOI['wall_proportions'][0], '.', color=color_1)
plt.plot(np.ones((5,))*2.8, gaze_AOI['ceiling_proportions'][0], '.', color=color_1)

plt.plot(np.ones((5,))*1.2, gaze_AOI['trampoline_bed_proportions'][1], '.', color=color_2)
plt.plot(np.ones((5,))*2.2, gaze_AOI['wall_proportions'][1], '.', color=color_2)
plt.plot(np.ones((5,))*3.2, gaze_AOI['ceiling_proportions'][1], '.', color=color_2)

xticks_names = ['Trampoline','Murs','Plafond']
plt.xticks(np.array([1, 2, 3]), xticks_names)
plt.ylabel("Répartition du regard [%]")
plt.ylim(0, 65)

plt.plot(np.array([0.8, 1.2]),
         np.array([np.max(gaze_AOI['trampoline_bed_proportions'])+3, np.max(gaze_AOI['trampoline_bed_proportions'])+3]),
         '-k')
plt.text(0.95, np.max(gaze_AOI['trampoline_bed_proportions'])+4, '**')

plt.plot(np.array([2.8, 3.2]),
         np.array([np.max(gaze_AOI['ceiling_proportions'])+3, np.max(gaze_AOI['ceiling_proportions'])+3]),
         '-k')
plt.text(2.95, np.max(gaze_AOI['ceiling_proportions'])+4, '**')

plt.savefig(root_path + "/Documents/Programmation/rectangle-labelling/rectangle_heatmap/Results_tempo/Proportions_GuSe_AlLe.png", dpi=300, format="png")
plt.show()



