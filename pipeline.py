import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from thresholding import thresholding_pipeline, region_of_interest

### Read test image
img = cv2.imread('test_images/test2.jpg')
#img = cv2.imread('camera_cal/test.jpg')

### Load camera calibration parameters
cal_params = pickle.load(open('camera_cal/dist_pickle.p', 'rb'))
mtx = cal_params["mtx"]
dist = cal_params["dist"]

### Test undistortion and save result
undistorted = cv2.undistort(img, mtx, dist, None, mtx)
undistorted_orig = undistorted.copy()
#cv2.imwrite('output_images/test_undist_in_pipeline.jpg',undistorted)
cv2.imshow('Undistorted Image', undistorted)
cv2.waitKey(1000)

### Perform thresholding
threshold_applied = thresholding_pipeline(undistorted)
cv2.imshow('Threshold applied', threshold_applied)
cv2.waitKey(1000)

### Apply ROI
vertices = np.array([[(0,720),(550+50, 420), (730-50, 420), (1280,720)]], dtype=np.int32)
roi_applied = region_of_interest(threshold_applied, vertices)
cv2.imshow('ROI applied', roi_applied)
cv2.waitKey(1000)

### Perform perspective transformation
show_lines = True
img_size = (roi_applied.shape[1], roi_applied.shape[0])

if show_lines == True:
    #img_size = (undistorted.shape[1], undistorted.shape[0])
    #print(img_size)
    undistorted = cv2.line(undistorted,(216,720),(595,450),(0,0,255),2)
    undistorted = cv2.line(undistorted,(1108,720),(689,450),(0,0,255),2)
    undistorted = cv2.line(undistorted,(595,450),(689,450),(0,0,255),2)
    #cv2.imwrite('output_images/annotated_lines_before_perspective_transform.jpg',undistorted)
    cv2.imshow('Annotated lines', undistorted)
    cv2.waitKey(1000)

src = np.float32([[595,450],[689,450],[216,720],[1108,720]])
#dst = np.float32([[216,0],[1108,0],[216,720],[1108S,720]])
dst = np.float32([[216+100,0],[1108-100,0],[216+100,720],[1108-100,720]]) #WARNING: 100 pixels of image shrinking need to be converted into the radius scale coeficient and applied later on
# Given src and dst points, calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)
# Warp the image using OpenCV warpPerspective()
warped = cv2.warpPerspective(roi_applied, M, img_size)
B_bin_channel = warped[:,:,0]
G_bin_channel = warped[:,:,1]
R_bin_channel = warped[:,:,2]
Merged_binary = np.zeros_like(G_bin_channel)
Merged_binary[(R_bin_channel > 0.0) | (G_bin_channel > 0.0) ] = 1
cv2.imshow('Merged_binary', Merged_binary)
cv2.waitKey(1000)
# Convert to B&W for demonstration purpose
Merged_binary_BW = np.zeros_like(Merged_binary)
Merged_binary_BW[(Merged_binary == 1)] = 255
#warped_orig = warped.copy()
#warped_orig = cv2.bitwise_not(warped_orig)
#cv2.imwrite('output_images/perspective_transform_applied_bin.jpg', Merged_binary)
#cv2.imwrite('output_images/perspective_transform_applied.jpg', Merged_binary_BW)

warped = cv2.line(warped,(216+100,0),(216+100,720),(255,0,0),5)
warped = cv2.line(warped,(1108-100,0),(1108-100,720),(255,0,0),5)
#cv2.imwrite('output_images/annotated_lines_after_perspective_transform.jpg',warped)
cv2.imshow('Warped image', warped)

### Find lines

cv2.waitKey()
cv2.destroyAllWindows()



