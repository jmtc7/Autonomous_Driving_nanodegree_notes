import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """ 
    Applies Sobel x or y, then takes an absolute value and applies a threshold.
    """
    
    # Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray, cv2.CV_64F, orient=='x', orient=='y', ksize=sobel_kernel)
    
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Create a mask of 1's where the scaled gradient magnitude
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1 


    # Return this mask as your binary_output image
    return grad_binary


def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    """ 
    applies Sobel x and y, then computes the magnitude of the gradient
    and applies a threshold
    """
    
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the magnitude
    mag = np.sqrt(sobelx**2 + sobely**2)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scale = np.max(mag)/255 
    scaled_mag = (mag/scale).astype(np.uint8) 
    
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(scaled_mag)
    binary_output[(scaled_mag > mag_thresh[0]) & (scaled_mag < mag_thresh[1])] = 1 

    # Return this mask as your binary_output image
    return binary_output

    
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Applies Sobel x and y, then computes the direction of the gradient
    and applies a threshold.
    """

    # Gradients in x and y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Absolute values of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # Calculate the direction of the gradient 
    direction = np.arctan2(abs_sobely, abs_sobelx)

    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    return binary_output



# Read the image and convert it to grayscale
image = mpimg.imread('signs_vehicles_xygrad.png')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Run the functions
gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(20, 100))
grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=3, thresh=(20, 100))
mag_binary = mag_thresh(gray, sobel_kernel=3, mag_thresh=(30, 100))
dir_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))

# Combine binary outputs
combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1


# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(combined, cmap='gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
