
import numpy as np
import cv2
import pickle

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def draw_lines(image, lines):
    for line in lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image, (x1, y1), (x2, y2), (0,0,255), 2)

def xy_to_rtheta(x0, x1, y0, y1):
    theta = np.arctan((x0 - x1) / (y1 - y0))
    r = x0 * np.cos(theta) + y0 * np.sin(theta)
    return r, theta

########################## Beginning of code ########################################3

file = open(f"../output/PI world v1 ps1_181_undistorted_images.pkl", "rb")
image_big = pickle.load(file)
ratio_image = 1.5
width, height, rgb = np.shape(image_big)
image = cv2.resize(image_big, (int(round(width / ratio_image)), int(round(height / ratio_image))))

Window_name = "Image"

orig = image.copy()
draw = image.copy()

vert_lines = np.zeros((5, 2))
vert_lines[0, :] = xy_to_rtheta(192, 303, 87, 628)
vert_lines[1, :] = xy_to_rtheta(260, 361, 66, 619)
vert_lines[2, :] = xy_to_rtheta(386, 397, 319, 406)
vert_lines[3, :] = xy_to_rtheta(417, 490, 13, 601)
vert_lines[4, :] = xy_to_rtheta(501, 552, 1, 599)

hori_lines = np.zeros((7, 2))
hori_lines[0, :] = xy_to_rtheta(192, 501, 87, 1)
hori_lines[1, :] = xy_to_rtheta(228, 519, 249, 179)
hori_lines[2, :] = xy_to_rtheta(307, 452, 308, 276)
hori_lines[3, :] = xy_to_rtheta(344, 435, 374, 355)
hori_lines[4, :] = xy_to_rtheta(333, 470, 444, 419)
hori_lines[5, :] = xy_to_rtheta(286, 541, 517, 479)
hori_lines[6, :] = xy_to_rtheta(303, 552, 628, 599)


test = np.argsort(np.abs(vert_lines[:, 0]))
vert_lines = vert_lines[test]

test = np.argsort(np.abs(hori_lines[:, 0]))
hori_lines = hori_lines[test]

# Finding the intersection points of the lines
points = []
num_vert_lines = vert_lines.shape[0]
num_hori_lines = hori_lines.shape[0]
for i in range(num_vert_lines):
    for j in range(num_hori_lines):
        point = intersection(vert_lines[i], hori_lines[j])
        points.append(point)

# Drawing the lines and points
draw_lines(draw, vert_lines)
draw_lines(draw, hori_lines)
[cv2.circle(draw, (p[0][0], p[0][1]), 5, 255, 2) for p in points]

# Finding the four corner points and ordering them
pts = np.asarray(points)
four_vertices = order_points(pts.reshape(pts.shape[0], pts.shape[2]))

# Giving a 5 pixels margin for better readability
four_vertices[0] = four_vertices[0][0] - 5, four_vertices[0][1] - 5
four_vertices[1] = four_vertices[1][0] + 5, four_vertices[1][1] - 5
four_vertices[2] = four_vertices[2][0] + 5, four_vertices[2][1] + 5
four_vertices[3] = four_vertices[3][0] - 5, four_vertices[3][1] + 5

# Perspective transform to get the warped image
warped = four_point_transform(orig, four_vertices)

# Displaying the results
cv2.imshow("Input", resize(image, height = 650))
cv2.imshow("Marks", resize(draw, height = 650))
cv2.imshow("Warped", resize(warped, height = 650))
cv2.waitKey(0)








