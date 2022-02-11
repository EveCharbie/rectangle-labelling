# rectangle-labelling
Python interface allowing to label the corners of a quadrangle, to counter the fisheye 
effect on videos, to distort an image to get a plane rectangle and to create a heatmap 
in the reference frame of the rectangle.
Strongly inspired from other github repos and internet sources, I tried to give them 
credits at the beginning of the files.


# To install:
conda create --name [name] python=3.9

conda activate [name]

conda install -c conda-forge scipy numpy opencv pytorch pickle tqmd


# Workflow:
1. Calibrate for fisheye lens with de-distort_fisheye/Fisheye_calibration.py -> saves matrix to Fisheye_KD_matrix.pkl
2. Dedistort image to correct for fisheye effect with de-distort_fisheye/Fisheye_corrections.py -> saves to [movie_file]_undistorted_images.pkl
3. Select the min/max in hsv of the 'red' color with  autodetection/color_selector.py -> saves to colors_selected.pkl
4. Find red lines on the trampoline with autodetection/find_lines.py-> saves to [movie_file]_lined_images.pkl
5. Manually label the corners of the trampoline with labeller/rectangle-labeler-video.py -> saves to [movie_file]_labeling_points.pkl
6. Distort the images to have a flat rectangle with distort_to_rectangle.py-> saves to -> saves to [movie_file]_flat_rectangle.pkl

TO DO:
- Clean code ! -> with output variables
- add the tracking of the pixel where the athlete si looking



