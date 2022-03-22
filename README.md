# rectangle-labelling
Python interface allowing to label key points on a trampoline, to counter the fisheye 
effect on videos, to distort an image to get a plane rectangle, to create a heatmap 
in the reference frame of the rectangle and to measure significant differences between the heatmaps (this part is in matlab).


# To install:
conda create --name [name] python=3.9

conda activate [name]

conda install -c conda-forge scipy numpy opencv pytorch pickle tqmd jupyter ipywidgets

Also need matlab and the fieldtrip toolbox (follow instruction here to request a version  https://www.fieldtriptoolbox.org/)

# Workflow:
1. Calibrate for fisheye lens with de-distort_fisheye/Fisheye_calibration.py -> saves matrix to Fisheye_KD_matrix.pkl
2. Dedistort image to correct for fisheye effect with de-distort_fisheye/Fisheye_corrections.py -> saves to [movie_file]_undistorted_images.pkl
3. Manually label the key points of the trampoline with labeller/rectangle-labeler-video-suplementay-info.py -> saves to [movie_file]_labeling_points.pkl
4. Get the gaze heatmaps with rectangle_heatmap/create_gaussian_heatmap.py-> saves to -> output/Results/[subject name]/[movement name]/[movie_file]_heat_map_[i].pkl
5. Get the significant differences between the heatmaps with Stats_diff_heatmaps.m -> saves to output/Stats_diffs/[group names]_[movement name]_stats_diffs.mat



