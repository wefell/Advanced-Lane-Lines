import cv2
import numpy as np
import glob
import pickle

# preprare object points
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, :6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# glob calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# draw corners
for index, filename in enumerate(images):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # if found add object points and image points
    if ret is True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # draw corners
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        write_name = './camera_cal/corners_found' + str(index) + '.jpg'
        cv2.imwrite(write_name, img)

# load reference image
img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                   img_size, None, None)

# save calibration
dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump(dist_pickle, open('./calibration_pickle.p', 'wb'))
