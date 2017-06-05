import pickle
import cv2
import numpy as np

# Test undistortion on an image
img = cv2.imread('test_images/straight_lines2.jpg')
#img = cv2.imread('camera_cal/test.jpg')

#load camera calibration parameters
cal_params = pickle.load(open('camera_cal/dist_pickle.p', 'rb'))
mtx = cal_params["mtx"]
dist = cal_params["dist"]

#test undistortion and save result
undistorted = cv2.undistort(img, mtx, dist, None, mtx)
undistorted_orig = undistorted.copy()
cv2.imwrite('output_images/test_undist_in_pipeline.jpg',undistorted)
cv2.imshow('Undistorted Image', undistorted)
cv2.waitKey(1000)

#perform perspective transformation
img_size = (undistorted.shape[1], undistorted.shape[0])
print(img_size)
undistorted = cv2.line(undistorted,(216,720),(595,450),(0,0,255),2)
undistorted = cv2.line(undistorted,(1108,720),(689,450),(0,0,255),2)
undistorted = cv2.line(undistorted,(595,450),(689,450),(0,0,255),2)
cv2.imwrite('output_images/annotated_lines_before_perspective_transform.jpg',undistorted)
cv2.imshow('Annotated lines', undistorted)
cv2.waitKey(10000)

src = np.float32([[595,450],[689,450],[216,720],[1108,720]])
dst = np.float32([[216,0],[1108,0],[216,720],[1108,720]])
# Given src and dst points, calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)
# Warp the image using OpenCV warpPerspective()
warped = cv2.warpPerspective(undistorted_orig, M, img_size)
warped_orig = warped.copy()

warped = cv2.line(warped,(216,0),(216,720),(0,0,255),5)
warped = cv2.line(warped,(1108,0),(1108,720),(0,0,255),5)
cv2.imwrite('output_images/annotated_lines_after_perspective_transform.jpg',warped)
cv2.imshow('Warped image', warped)


cv2.waitKey()

cv2.destroyAllWindows()
