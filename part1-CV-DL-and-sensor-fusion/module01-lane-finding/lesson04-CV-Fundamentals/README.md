# Computer Vision fundamentals for lane detection

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Color selection
We can transform an RGB image to a binary one in which only the pixels whose values in each color channel are higher than certain thresholds. For example, requiring them to be higher than 200 (with values between 0 and 255) will output an image in which only the pixels with a color close to white will be activated.

## Region of Interest (RoI)
By knowing where in the car the camera is mounted, we can *crop* the image in order to only take into account the pixels activated by the color selection that are in this region. The region will be one in which the lane lines will be contained in every frame (at least the closest part of them to the car).

## Trial code: lane-detection.py
This code combines these two methods to select only the white pixels inside the RoI (shown in red when the script is executed).


## Canny edge detection
We can detect edges by converting an RGB image to grayscale and computing its gradients (derivate the 2D function represented by the image). This will provide higher (whiter) values in the places where the gray changes faster (higher contrast). This provides us with thick edges that are made thinner by the Canny algorithm, which will activate only the pixels that correspond to the strongest gradients.

As explained [here](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html), if the pixel gradient is higher than the upper threshold, it will be accepted. If it is under the lower one, it will be rejected. If it is in between, it will only be accepted if it is connected to a pixel over the upper threshold.

It is common to use 1/2 or 1/3 proportion between the low and high thresholds of the Canny algorithm. Another common practice is to apply a Gaussian blurr before the Canny algorithm. This helps to remove noise, reducing the false edges.

## Hough transform to find lines given the edges
Given a 2D space, we can represent a line as the function *y=mx+b*. If we build a 2D using *m* and *b* instead of *x* and *y*, a line in an *XY* 2D space will be a point. This new 2D space is known as the Hough Space.

This way, two paralel lines will have the same *m* but different *b*, so in Hough space, these lines will be represented by two dots vertically aligned. Moreover, a point in an *XY* space can be represented in infinite ways in Hough space. i.e. it will be represented by a line. When having two points in an *XY* space, they will have a 2 lines representation in Hough space that will intersect in a certain point. This intersection will be the line that connects the two points.

Given this *intersection trick*, we can use points in the *XY* space (image), convert them to Hough space, search for things close to be multiple lines intersecting in one point and this point will be the line that (kind of) connects several points in *XY* space.

Once having the points associated to edges (thanks to the Canny algorithm), we can fordward these points into a Hough transformation to search for intersections and, therefore, lines that connect these Canny-obtained sparse points.

When having several points, such as a *sampling* of a square, we will have certain points in Hough space with more intersections, that will be the ones connecting more points (the vertex of the square connect all the points in the edges of it) and some other intersections that will connect less points (each point with each other, such as de diagonals).
