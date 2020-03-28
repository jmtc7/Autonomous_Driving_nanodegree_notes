#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read in an Image

# In[2]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[3]:


import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=12):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    # Get the y size of the image (FLIPPED AXIS)
    ysize = img.shape[0]
    xsize = img.shape[1]
    
    # List of slopes and biases of the left and right limits of the lane (need individual initializations)
    l_slopes = []
    l_biases = []
    r_slopes = []
    r_biases = []

    # Iterate trough the lines and their components
    #Try-catch to avoid stops casused by empty "lines" lists
    try:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (x1-x2)/(y1-y2)
                bias = x1-(slope*y1)

                # If the slope is positive, it belongs to the right limit. If negative, left limit
                if slope < -0.3 and bias < 1000:
                    l_slopes.append(slope)
                    l_biases.append(bias)
                elif slope > 0.3 and bias > -100:
                    r_slopes.append(slope)
                    r_biases.append(bias)
                else:
                    pass #If slope close to horizontal, do not take this line into account
    except:
        print('[!] Warning! No lines detected for this frame.')
        
    #Try-catch to avoid stops casused by operations with infinite values
    try:
        # Average equation params for lane limits  
        l_slope_avg = np.array(l_slopes).mean()
        l_bias_avg = np.array(l_biases).mean()
        r_slope_avg = np.array(r_slopes).mean()
        r_bias_avg = np.array(r_biases).mean()

        # Compute min and max point inside mask
        l_y1 = r_y1 = xsize # Bottom of the image as bottom limit
        l_y2 = r_y2 = int(ysize*0.6) # 60% of the image as top limit (empiric and visual choice)
        
        l_x1 = int((l_y1 * l_slope_avg + l_bias_avg)) # x = my+b with FLIPPED AXIS
        l_x2 = int((l_y2 * l_slope_avg + l_bias_avg))
        
        r_x1 = int((r_y1 * r_slope_avg + r_bias_avg))
        r_x2 = int((r_y2 * r_slope_avg + r_bias_avg))
    
        # Draw average lane lines (STANDARD IMAGE AXIS)
        cv2.line(img, (l_x1, l_y1), (l_x2, l_y2), color, thickness)
        cv2.line(img, (r_x1, r_y1), (r_x2, r_y2), color, thickness)
    except:
        print('[!] Warning! An operation with infinite value occured.')
    

    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:


import os
os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[5]:


#GENERAL SETUP
image_list = os.listdir("test_images/") #List with the names of the images
output_images = [] #List with the output images (will be filled in the loop)


#Iterate trough the image names
for image_name in image_list:
    
    # [STEP 0]: READ IMAGE
    # Read in the image
    image = cv2.imread('test_images/'+image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB color space
    
    # Grab the x and y size of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    
    
    # [STEP 1]: COLOR THRESHOLDING
    # Get yellow mask
    hsv_input = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # Convert to HSV color space
    lower_yell_thresh = np.array([15, 100, 175]) # Lower threshold to get yellow
    upper_yell_thresh = np.array([45, 255, 255]) # Upper yellow thresh
    yellow_mask = cv2.inRange(hsv_input, lower_yell_thresh, upper_yell_thresh) #Mask with yellow pixels
    
    # Get white mask
    lower_whit_thresh = np.array([0, 0, 225]) # Lower white thresh
    upper_whit_thresh = np.array([179, 25, 255]) # Upper white thresh
    white_mask = cv2.inRange(hsv_input, lower_whit_thresh, upper_whit_thresh) # Mask with white 
    
    # Compute combined mask
    color_mask = white_mask | yellow_mask
    
    # Dilate mask to consider more pixels while computing the gradients while applying Canny
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    dil_color_mask = cv2.dilate(color_mask, kernel) # Dilated color mask
    
    # Apply dilated mask to color image
    color_out = cv2.bitwise_and(image, image, mask=dil_color_mask)

    
    # [STEP 2]: APPLY A MASK WITH THE ROI
    roi_vertices = np.array([[(0,ysize), (0.45*xsize,ysize/2), (0.55*xsize,ysize/2), (xsize,ysize)]], dtype=np.int32)
    roi_color = region_of_interest(color_out, roi_vertices)    

    
    # [STEP 3]: CANNY EDGE DETECTION
    # Convert the color activation to grayscale
    gray = grayscale(roi_color)

    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, 5)

    # Define our parameters for Canny and apply
    edges = canny(blur_gray, 75, 150)

    
    # [STEP 4]: SEARCH LINES WITH HOUGH TRANSFORM
    lines = hough_lines(edges, 2, np.pi/180, 25, 5, 5)

    
    # [STEP 5]: Merge lines with original image and append to the list of output images
    proc_image = weighted_img(lines, image)
    output_images.append(proc_image)

    
# Show the processed images (in two columns)
fig = plt.figure(figsize=(15, 15))
x_dim = 2
y_dim = len(image_list)//x_dim

for idx_y in range(y_dim):
    for idx_x in range(x_dim):
        #Plot color image with solution in red
        plotted_imgs = x_dim*idx_y+idx_x #Amount of plotted images
        a = fig.add_subplot(y_dim, x_dim, plotted_imgs+1)
        a.set_title(image_list[plotted_imgs])
        plt.imshow(output_images[plotted_imgs])
        plt.axis('off')


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[6]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[7]:


def process_image(image):

    # Grab the x and y size of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    
    
    # [STEP 1]: COLOR THRESHOLDING
    # Get yellow mask
    hsv_input = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # Convert to HSV color space
    lower_yell_thresh = np.array([15, 100, 175]) # Lower threshold to get yellow
    upper_yell_thresh = np.array([45, 255, 255]) # Upper yellow thresh
    yellow_mask = cv2.inRange(hsv_input, lower_yell_thresh, upper_yell_thresh) #Mask with yellow pixels
    
    # Get white mask
    lower_whit_thresh = np.array([0, 0, 225]) # Lower white thresh
    upper_whit_thresh = np.array([179, 25, 255]) # Upper white thresh
    white_mask = cv2.inRange(hsv_input, lower_whit_thresh, upper_whit_thresh) # Mask with white 
    
    # Compute combined mask
    color_mask = white_mask | yellow_mask
    
    # Dilate mask to consider more pixels while computing the gradients while applying Canny
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    dil_color_mask = cv2.dilate(color_mask, kernel) # Dilated color mask
    
    # Apply dilated mask to color image
    color_out = cv2.bitwise_and(image, image, mask=dil_color_mask)

    
    # [STEP 2]: APPLY A MASK WITH THE ROI
    roi_vertices = np.array([[(0,ysize), (0.45*xsize,ysize/2), (0.55*xsize,ysize/2), (xsize,ysize)]], dtype=np.int32)
    roi_color = region_of_interest(color_out, roi_vertices)    

    
    # [STEP 3]: CANNY EDGE DETECTION
    # Convert the color activation to grayscale
    gray = grayscale(roi_color)

    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, 5)

    # Define our parameters for Canny and apply
    edges = canny(blur_gray, 75, 150)

    
    # [STEP 4]: SEARCH LINES WITH HOUGH TRANSFORM
    lines = hough_lines(edges, 2, np.pi/180, 25, 5, 5)

    
    # [STEP 5]: Merge lines with original image and append to the list of output images
    proc_image = weighted_img(lines, image)
    
    return proc_image


# Let's try the one with the solid white lane on the right first ...

# In[8]:


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[9]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[10]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')


# In[11]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[12]:


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[13]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


# In[ ]:




