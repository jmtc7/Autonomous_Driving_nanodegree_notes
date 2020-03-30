import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Undistort image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    
    # If success finding corners
    if ret:
        # Draw found corners
        img = cv2.drawChessboardCorners(undist, (nx,ny), corners, ret)
        
        # Define 4 source points (a rectangle formed the 4 more external corners)
        idx1 = 0 # Top left corner
        idx2 = nx-1 # Top right
        idx3 = nx*ny-1 # Bottom right
        idx4 = nx*(ny-1) # Bottom left

        src = np.float32([corners[idx1], corners[idx2], corners[idx3], corners[idx4]])
        
        # Define 4 destination points (image vertices)
        ysize = undist.shape[0]
        xsize = undist.shape[1]
        
        dst = np.float32([[100,100],[xsize-100,100],[xsize-100,ysize-100],[100,ysize-100]])
        
        # Get the transformation matrix
        M = cv2.getPerspectiveTransform(src, dst)
        
        # Get top-down view
        img_size = (xsize, ysize)
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
        
    else:
        M=None
        warped = undist

    return warped, M


# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # Number of inside corners in X
ny = 6 # Y corners 

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
