import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from thresholding import thresholding_pipeline, region_of_interest
from Line import Line
from moviepy.editor import VideoFileClip

left_line = Line()
right_line = Line()


clip = VideoFileClip("project_video.mp4")
def running_average(buffer, current_fit):
    average = np.array([0,0,0], dtype='float')
    for i in range(len(buffer)):
        average += buffer[i]
    average = average/len(buffer)
    return average

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
    #dst = np.float32([[216,0],[1108,0],[216,720],[1108,720]])
    dst = np.float32([[216+100,0],[1108-100,0],[216+100,720],[1108-100,720]]) #WARNING: 100 pixels of image shrinking need to be converted into the radius scale coeficient and applied later on
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(roi_applied, M, img_size)
    B_bin_channel = warped[:,:,0]
    G_bin_channel = warped[:,:,1]
    R_bin_channel = warped[:,:,2]
    merged_binary = np.zeros_like(G_bin_channel)
    merged_binary[(R_bin_channel > 0.0) | (G_bin_channel > 0.0) | (B_bin_channel > 0.0)] = 1
    #!cv2.imshow('Merged_binary', merged_binary)
    #!cv2.waitKey(1000)
    # Convert to B&W for demonstration purpose
    #merged_binary_BW = np.zeros_like(merged_binary)
    #merged_binary_BW[(merged_binary == 1)] = 255
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
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Processing first frame or if detection failed
    if (left_line.detected == False) & (right_line.detected == False):

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
        # Chose minimum number of windows with good indicies for polynomial fitting
        min_windows = 7
    
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
                left_line.new_windows_cnt +=1
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                right_line.new_windows_cnt +=1
    
        if (left_line.new_windows_cnt >= min_windows) & (right_line.new_windows_cnt >= min_windows):
    
            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit a second order polynomial to each
            left_line.current_fit = np.polyfit(lefty, leftx, 2)
            right_line.current_fit = np.polyfit(righty, rightx, 2)
        
            #Visualization
            #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
            # Do not do window search anymore
            left_line.detected = True
            right_line.detected = True
            left_line.new_windows_cnt = 0
            right_line.new_windows_cnt = 0
                
            
    else:
        
        ### Process next frame without window search
        nonzero = merged_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_line.best_fit[0]*(nonzeroy**2) + left_line.best_fit[1]*nonzeroy + left_line.best_fit[2] - margin)) & (nonzerox < (left_line.best_fit[0]*(nonzeroy**2) + left_line.best_fit[1]*nonzeroy + left_line.best_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_line.best_fit[0]*(nonzeroy**2) + right_line.best_fit[1]*nonzeroy + right_line.best_fit[2] - margin)) & (nonzerox < (right_line.best_fit[0]*(nonzeroy**2) + right_line.best_fit[1]*nonzeroy + right_line.best_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_line.current_fit = np.polyfit(lefty, leftx, 2)
        right_line.current_fit = np.polyfit(righty, rightx, 2)

        ### Next frame visualization
        # Create an image to draw on and an image to show the selection window
        window_img = np.zeros_like(out_img)
        # Generate x and y values for plotting
        ploty = np.linspace(0, merged_binary.shape[0]-1, merged_binary.shape[0] )
        left_fitx = left_line.current_fit[0]*ploty**2 + left_line.current_fit[1]*ploty + left_line.current_fit[2]
        right_fitx = right_line.current_fit[0]*ploty**2 + right_line.current_fit[1]*ploty + right_line.current_fit[2]
        
        if (left_fitx is None or right_fitx is None):
            left_line.detected = False
            right_line.detected = False
        
        # Color in left and right line pixels
        #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        #left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        #left_line_pts = np.hstack((left_line_window1, left_line_window2))
        #right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        #right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        #out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)


    #filter polynomials
    left_line.buffer.append(left_line.current_fit)
    right_line.buffer.append(right_line.current_fit)
    left_line.best_fit = running_average(left_line.buffer, left_line.current_fit)
    right_line.best_fit = running_average(right_line.buffer, right_line.current_fit)


    # Define conversions in x and y from pixels space to meters
    x_shrink_factor = 1.15
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/(700*x_shrink_factor) # meters per pixel in x dimension
    left_x_scaled = list(map(lambda x: x*xm_per_pix, leftx))
    left_y_scaled = list(map(lambda x: x*ym_per_pix, lefty))
    right_x_scaled = list(map(lambda x: x*xm_per_pix, rightx))
    right_y_scaled = list(map(lambda x: x*ym_per_pix, righty))

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(left_y_scaled, left_x_scaled, 2)
    right_fit_cr = np.polyfit(right_y_scaled, right_x_scaled, 2)

    # Calculate the new radii of curvature
    y_eval = 719 #maximum y-value corresponding to the bottom of the image
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # compute the offset from the center
    left = np.mean(leftx[len(leftx)//2:len(leftx)-1])
    right = np.mean(rightx[len(rightx)//2:len(rightx)-1])
    center_offset_mtrs = (merged_binary.shape[1]/2-np.mean([left, right]))*xm_per_pix/x_shrink_factor

    #Unwarping image
    out_img = np.dstack((merged_binary, merged_binary, merged_binary))*255
    y_points = np.linspace(0, merged_binary.shape[0]-1, merged_binary.shape[0])
    left_fitx = left_line.best_fit[0]*y_points**2 + left_line.best_fit[1]*y_points + left_line.best_fit[2]
    right_fitx = right_line.best_fit[0]*y_points**2 + right_line.best_fit[1]*y_points + right_line.best_fit[2]
    
    left_line_window = np.array(np.transpose(np.vstack([left_fitx, y_points])))
    right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_fitx, y_points]))))
    line_points = np.vstack((left_line_window, right_line_window))
    cv2.fillPoly(out_img, np.int_([line_points]), [0,255, 0])

    M_inv = cv2.getPerspectiveTransform(dst,src)

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    unwarped = cv2.warpPerspective(out_img, M_inv, img_size , flags=cv2.INTER_LINEAR)

    undistorted = undistorted.astype(float)
    result = cv2.addWeighted(undistorted, 1, unwarped, 0.3, 0)

    #Annotate image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,"left_curverad {0:4.1f} right_curverad {1:4.1f}".format(left_curverad, right_curverad),(300,100), font, 1,(255,255,255),2)
    cv2.putText(result,"center offset {0:4.1f} mtrs".format(center_offset_mtrs),(450,150), font, 1,(255,255,255),2)

    return result

new_clip = clip.fl_image( process_image )
new_clip.write_videofile("project_video_processed.mp4", audio=False)

### Test on images
#img = cv2.imread('test_images/test2.jpg')
#process_image(img)
#process_image(img)
#process_image(img)

#cv2.waitKey()
#cv2.destroyAllWindows()



