import cv2
from matplotlib import pyplot as plt
import numpy as np

# Edit this function to create your own pipeline.
def thresholding_pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    #s_channel_test = np.copy(s_channel)
    #binary_output = np.zeros_like(s_channel_test)
    #binary_output[(s_channel_test > 150) & (s_channel_test <= 255)] = 1
    #cv2.imshow('s_channel', binary_output)
    
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
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
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

#test functions
enable_test = False

if enable_test == True:
    
    image = cv2.imread('test_images/test4.jpg')
    #image = cv2.undistort(image, mtx, dist, None, mtx)

    threshold_applied = thresholding_pipeline(image)
    cv2.imshow('Threshold applied', threshold_applied)

    #Define region of interest
    #imshape = image.shape #720 x 1280
    vertices = np.array([[(0,720),(550, 420), (730, 420), (1280,720)]], dtype=np.int32)
    roi_applied = region_of_interest(threshold_applied, vertices)
    cv2.imshow('ROI applied', roi_applied)

    #annotated = cv2.line(roi_applied,(216,720),(595,450),(0,0,255),2)
    #annotated = cv2.line(annotated,(1108,720),(689,450),(0,0,255),2)
    #annotated = cv2.line(annotated,(595,450),(689,450),(0,0,255),2)
    #cv2.imshow('Annotated lines', annotated)

    cv2.waitKey()

    cv2.destroyAllWindows()
