import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from thresholding import thresholding_pipeline, region_of_interest
from Line import Line

left_line = Line()
right_line = Line()

from moviepy.editor import VideoFileClip # just import what you need

clip = VideoFileClip("project_video.mp4")
#clip.preview(fps=15, audio=False)


### Read test image
#img = cv2.imread('test_images/test2.jpg')
#img = cv2.imread('camera_cal/test.jpg')

def process_image(image):
    ### Load camera calibration parameters
    cal_params = pickle.load(open('camera_cal/dist_pickle.p', 'rb'))
    mtx = cal_params["mtx"]
    dist = cal_params["dist"]

    ### Undistort image
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    #undistorted_orig = undistorted.copy()
    #cv2.imwrite('output_images/test_undist_in_pipeline.jpg',undistorted)
    #!cv2.imshow('Undistorted Image', undistorted)
    #!cv2.waitKey(1000)

    ### Perform thresholding
    threshold_applied = thresholding_pipeline(undistorted)
    #!cv2.imshow('Threshold applied', threshold_applied)
    #!cv2.waitKey(1000)

    ### Apply ROI
    vertices = np.array([[(0,720),(550+50, 420), (730-50, 420), (1280,720)]], dtype=np.int32)
    roi_applied = region_of_interest(threshold_applied, vertices)
    #!cv2.imshow('ROI applied', roi_applied)
    #!cv2.waitKey(1000)

    ### Perform perspective transformation
    show_lines = False
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
    merged_binary = np.zeros_like(G_bin_channel)
    merged_binary[(R_bin_channel > 0.0) | (G_bin_channel > 0.0) ] = 1
    #!cv2.imshow('Merged_binary', merged_binary)
    #!cv2.waitKey(1000)
    # Convert to B&W for demonstration purpose
    merged_binary_BW = np.zeros_like(merged_binary)
    merged_binary_BW[(merged_binary == 1)] = 255
    #warped_orig = warped.copy()
    #warped_orig = cv2.bitwise_not(warped_orig)
    #cv2.imwrite('output_images/perspective_transform_applied_bin.jpg', merged_binary)
    #cv2.imwrite('output_images/perspective_transform_applied.jpg', merged_binary_BW)

    if show_lines == True:
        warped = cv2.line(warped,(216+100,0),(216+100,720),(255,0,0),5)
        warped = cv2.line(warped,(1108-100,0),(1108-100,720),(255,0,0),5)
        #cv2.imwrite('output_images/annotated_lines_after_perspective_transform.jpg',warped)
        cv2.imshow('Warped image', warped)

    # Take a histogram of the bottom half of the image
    histogram = np.sum(merged_binary[merged_binary.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((merged_binary, merged_binary, merged_binary))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(merged_binary.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = merged_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = merged_binary.shape[0] - (window+1)*window_height
        win_y_high = merged_binary.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return out_img

new_clip = clip.fl_image( process_image )
new_clip.write_videofile("processed.mp4", audio=False)

### Find lines

#cv2.waitKey()
#cv2.destroyAllWindows()



