import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def hist(img):
    # Get X and Y sizes of the image
    ysize = img.shape[0]
    xsize = img.shape[1]
    half_ysize_int = ysize//2
    
    # Grab the bottom half of the image (lane lines will be more vertical near to the car)
    bottom_half = img[half_ysize_int:, :]
    
    # Sum across image pixels vertically
    histogram = np.array([]) # List to store how many white points are in the vertical of each X
    x_sum = 0 # Container to store the count of white points
    
    # Iterate columns of the half-image (histogram)
    for x in range(xsize): 
        # Iterate rows of the half-image (vertical)
        for y in range(half_ysize_int):  
            # If the pixel is in white, add it to the count
            if bottom_half[y,x] == 1:
                x_sum += 1
        histogram = np.append(histogram, [x_sum])
        x_sum = 0
    
    return histogram

# Load our image (and normalize from 0-255 to 0-1)
img = mpimg.imread('warped_example.jpg')/255

# Create histogram of image binary activations
histogram = hist(img)

# Visualize the resulting histogram
plt.plot(histogram)

plt.show()
