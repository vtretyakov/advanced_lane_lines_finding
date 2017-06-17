import cv2
from matplotlib import pyplot as plt
import numpy as np
import pickle
from moviepy.editor import VideoFileClip

#test functions
enable_test = False

clip = VideoFileClip("project_video.mp4").subclip(38,43)#.subclip(20,25)

def thresholding_pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100), g_thresh=(200, 255), l_thresh=(180, 255)):
    img = np.copy(img)
    img2 = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float)
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    mean_s = np.mean(s_binary)
    # Skip the channel if it has an abnormal dominance
    if mean_s > 0.04:
        s_binary = np.zeros_like(s_channel)
    g_channel = rgb[:,:,1]
    r_channel = rgb[:,:,0]
    g_channel_binary = np.zeros_like(g_channel)
    r_channel_binary = np.zeros_like(r_channel)
    g_channel_binary[((g_channel > g_thresh[0]) & (g_channel <= g_thresh[1]))] = 1
    g_channel_binary[0:g_channel_binary.shape[0]//2, 0:g_channel_binary.shape[1]] = 0
    mean_g = np.mean(g_channel_binary)
    # Skip the channel if it has an abnormal dominance
    if mean_g > 0.04:
        g_channel_binary = np.zeros_like(g_channel)
    l_channel_binary = np.zeros_like(l_channel)
    l_channel_binary[((l_channel > l_thresh[0]) & (l_channel <= l_thresh[1]))] = 1
    l_channel_binary[0:l_channel_binary.shape[0]//2, 0:l_channel_binary.shape[1]] = 0
    l_channel_binary[l_channel_binary.shape[0]//2:l_channel_binary.shape[0], 0:l_channel_binary.shape[1]//5] = 0
    l_channel_binary[l_channel_binary.shape[0]//2:l_channel_binary.shape[0], l_channel_binary.shape[1] - l_channel_binary.shape[1]//5:l_channel_binary.shape[1]] = 0
    mean_l = np.mean(l_channel_binary)
    # Skip the channel if it has an abnormal dominance
    if mean_l > 0.01:
        l_channel_binary = np.zeros_like(l_channel)
    r_channel_binary[((r_channel > 220) & (r_channel <= 255))] = 1
    r_channel_binary[0:r_channel_binary.shape[0]//2, 0:r_channel_binary.shape[1]] = 0
    r_channel_binary[r_channel_binary.shape[0]//2:r_channel_binary.shape[0], r_channel_binary.shape[1] - r_channel_binary.shape[1]//5:r_channel_binary.shape[1]] = 0
    mean_r = np.mean(r_channel_binary)
    # Skip the channel if it has an abnormal dominance
    if mean_r > 0.01:
        r_channel_binary = np.zeros_like(r_channel)
    
    # Stack each channel
    combined_rg_and_l = np.zeros_like(g_channel_binary)
    combined_rg_and_l[(g_channel_binary > 0.0) | (l_channel_binary > 0.0) | (r_channel_binary > 0.0)] = 1
    color_binary = np.dstack(( combined_rg_and_l, sxbinary, s_binary))
    if enable_test == True:
        return color_binary*255
    else:
        return color_binary

def region_of_interest(img, vertices):
    """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (1,) * channel_count
    else:
        ignore_mask_color = 1
    
    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image




if enable_test == True:
    
    image = cv2.imread('test_images/test23.jpg')
    
    #Load camera calibration parameters
    cal_params = pickle.load(open('camera_cal/dist_pickle.p', 'rb'))
    mtx = cal_params["mtx"]
    dist = cal_params["dist"]
    
    #Apply calibration
    image = cv2.undistort(image, mtx, dist, None, mtx)

    #Apply thresholding
    threshold_applied = thresholding_pipeline(image)

    #Define and apply region of interest
    #imshape = image.shape #720 x 1280
    vertices = np.array([[(0,720),(550, 420), (730, 420), (1280,720)]], dtype=np.int32)
    roi_applied = region_of_interest(threshold_applied, vertices)

    #Display results
    #cv2.imwrite('output_images/threshold_applied_with_g_t4.jpg',threshold_applied*255)
    cv2.imshow('Threshold applied', threshold_applied)
    #cv2.imwrite('output_images/roi_applied.jpg',roi_applied*255)
    #cv2.imshow('ROI applied', roi_applied)

    annotated = cv2.line(roi_applied,(216,720),(595,450),(0,0,255),2)
    annotated = cv2.line(annotated,(1108,720),(689,450),(0,0,255),2)
    annotated = cv2.line(annotated,(595,450),(689,450),(0,0,255),2)
    #cv2.imshow('Annotated lines', annotated)

    cv2.waitKey()
    cv2.destroyAllWindows()
    
    new_clip = clip.fl_image( thresholding_pipeline )
    new_clip.write_videofile("project_video_processed.mp4", audio=False)
