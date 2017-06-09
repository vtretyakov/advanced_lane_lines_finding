import cv2
from matplotlib import pyplot as plt
import numpy as np

#color and gradient thresholding
image = cv2.imread('test_images/test4.jpg')
#image = cv2.undistort(image, mtx, dist, None, mtx)

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

result = thresholding_pipeline(image)

# Show the result
#print(result.shape)
#plt.subplot(231),plt.imshow(result,cmap='gray', extent=[0,1280,0,720], aspect=1),plt.title('ORIGINAL')
#plt.subplot(232),plt.imshow(result,cmap='gray', extent=[0,1280,0,720], aspect=1),plt.title('REPLICATE')

#plt.show()

cv2.imshow('Pipeline Result', result)

cv2.waitKey()


cv2.destroyAllWindows()
