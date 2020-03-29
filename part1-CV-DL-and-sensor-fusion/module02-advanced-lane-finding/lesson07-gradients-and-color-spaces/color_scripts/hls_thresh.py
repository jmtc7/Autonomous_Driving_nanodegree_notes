import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def hls_select(img, thresh=(0, 255)):
    """
    Thresholds the S-channel of HLS.
    """
    
    # Convert to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # Apply a threshold to the S channel
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    
    # Return a binary image of threshold result
    return binary_output
    
    
# Read in an image, you can also try test1.jpg or test4.jpg
image = mpimg.imread('test6.jpg')

# Optional TODO - tune the threshold to try to match the above image!    
hls_binary = hls_select(image, thresh=(150, 255))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.show()
