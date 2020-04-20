# Object Detection

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The aim of this lesson is to be able to study the traditional computer vision approaches to detect objects, which are based in using features to distinguish the target objects from the rest of the objects in the image and the background. This can be applied to each frame of a video to track relevant objects such as cars. To indicate the detections, *bounding boxes* can be used. In in the script `00_draw_bboxes.py`, a function that draws the given BBs is implemented. 


## Feature Extraction and Comparison
Color and gradient-based features have already been studied during this course. Each of them (and many others) are useful in one or another situations. This is why it is common to combine several features to achieve successful results. It is possible to use the pixels' intensity to get information about shapes and color, color histograms to get information about color and gradients to get information about shapes.

### Color-Based Features
A method for detecting objects in certain regions of a picture is ***Template Matching***. This consist in having a template object (e.g. a picture of an *ideal* car) and compare it to a region of the image using the extracted features of both of them. Some ways of doing it are:

- Substracting the color values, aggregating the differences and checking if the absolute value of the result is under a threshold.
- Check if the correlation between the template and the image region is high.

This method works in limited classes but is not really useful for self-driving cars. Anyway, a sample implementation can be found in the `01_template_match.py` script, which uses the list of templates in the `./cutouts` folder. It can be seen that it works properly, but whenever another image is used, it does not work because the shape, color or sizes of the cars will change. This method will only be useful whenever the objects to detect will not change much, such as logos or emojis.

Another option are the ***Histograms of Color***. Comparing the color histogram of a known object with the one of a region of the image, close color distributions will match. This makes the detection robuts to the pixel order, which means that the object will be detected even if it has a slightly different aspect or orientation. However, this will introduce several false positives. In `02_color_histogram.py`, the color histograms of one of the previously used cutouts are extracted (per each channel).

### Shape-Based Features
This will still have the problem of detect objects of the same class (e.g. car) of different colors. Here is where ***color spaces*** come into play. Some of these, such as HLS or LUV, have already been studied. This can be useful because cars are usually more saturated in color while the background tends to have more pale and neutral colors. In HSV, the car pixels seems to cluster well on the Saturation-Value (S and V channels) plane.

In order to choose a proper color space to get features from, it is useful to represent images in spaces where each channel of the color space is one dimension. If several images are represented in several color spaces, it is possible that we may find some plane, hyperplane or full space in which the pixels of the cars cluster nicely together. This can be done using the `03_color_spaces.py` script, which can be tested against the images in the `color_spaces_samples` folder (taken from the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)).

Even *template matching* is not the most appropiate method for the detection problem in self-driving cars, the color intensity is still a useful feature to be considered. Even the three channels at full resoulutions would be too much to be included in a feature vector, it is possible to use ***spatial binning*** to reduce the resolution of the image. An easy way of doing it is the OpenCV's `cv2.resize()` function. This is combined with a color space transformation in order to transform an input image to a feature vector in the `04_spatial_binnning.py` script.

### HOG Features
So far we have managed to properly use color information. However, cars can be of very different colors. This is why considering their shape is useful. This can be done using ***gradient features***. The main problem of this is that it makes the resulting feature vector too sensitive to gradient variations. The gradient-based feature vectors are obtained by computing the gradient of an image to use them to compute the magnitudes and, finally, the directions of the detected edges. These directions can be arranged in a vector instead of in a matrix to *define* the analyzed object.

As a more detailed explanation about how to compute a gradient-based feature vector, given a 64x64 image of a car, it is possible to compute the gradients, magnitudes and directions of each of its pixels and then divide the image in small 8x8 cells. Next, the histogram of gradien directions of each cell will be computed. Working at this small scale, usually dividing the angles between -180 and 180 degrees in 9 slots will be enaugh. Finally, the direction with the higher value of the histogram (orientation with highest accumulated magnitude) will be the **dominant gradient direction** of the 8x8 cell. Once having the dominant gradient directions of each cell, it will be possible to arrange them in a more compact vector to define the shape contained in the image. This method is known as **Histogram of Oriented Gradients** or HOG feature.

This, given the 8x8 cells, will be more robust to small variations in the sape while keeping the overall definition representative enaugh. This can be tweaked by adjusting the cell sizes, number of cells, overlapping between cells, etc. It is also possible to normalize the intensity accross small cell blocks.

[Here](https://www.youtube.com/watch?v=7S5qXET179I) is a presentation on using Histogram of Oriented Gradient (HOG) features for pedestrian detection by Navneet Dalal, the original developer of HOG for object detection. You can find his original paper on the subject [here](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf). 


NOTE: From now on, a *small* dataset will be used. It is contained in the `dataset` folder and is composed out of images from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/) and samples from the project video itself.

### Histograms
- Histograms of Color
- Histogram Comparison

### Color Spaces

### Gradient Features

### HOG Features

### Feature Combination and Normalization



## Classifier Building

### Labeled Data
### Data Preparation
### Training
### Parameter Tuning

### Color-Based Classifier
### HOG-Based Classifier


## Sliding Windows
- Ammount of windows
- Implementation
- Multi-scale windows
- Searching and classifying
  - HOG sub-sampling window classifier


## Other Considerations
### Multiple Detections and False Positives
### Tracking Pipeline
### Traditional VS Deep Learning Approaches





