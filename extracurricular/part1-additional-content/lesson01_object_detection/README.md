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

NOTE: From now on, a *small* dataset will be used. It is contained in the `dataset` folder and is composed out of images from the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/) and samples from the project video itself. Subsets of the dataset can be downloaded from these links: [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip) and the full dataset can be accessed from these other ones: [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). For reference, here is the link to the [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) in GitHub. Every Udacity dataset comes with a `labels.csv` file containing the bounding box corners of each labeled object. In the `05_car_notcar.py` script, the data can be classified as images containing a car or a non-car object based on the file names. It also generates a dictionary with the image shapes and types.

The [sci-image](http://scikit-image.org/) package contains a function to extract HOG features. A tutorial and explanation of it can be found [here](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html). It allows its user to set the number of `orientations` to be in the histogram (usually between 6 and 12), the `pixels_per_cell` (8x8 in the used example), the `cells_per_block` to define how many cells will form a block for the normalization, which is not necessary but usually leads to more robuts results and the `transform_sqrt` flag, which will activate the *power law* or *gamma normalization*. 

Given `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, and `orientations=9`, the straightforward assumption is to guess that the resulting feature vector will be *9×8×8=576* elements long. However, when using block normalization, the HOG features for all cells in each block are computed at each block position and the block steps across and down through the image cell by cell. Therefore, the actual ammount of elements will be *7×7×2×2×9=1764*.

It is possible to set the `feature_vector=True` flag when calling the `hog()` function to automatically unroll the features. The `06_get_hog.py` script implements a function that accepts argumetns to choose if we want the feature vector unrolled and a visualization that shows the dominant gradient direction within each cell with brightness corresponding to the strength of gradients in that cell.

### Feature combination
So far, it is clear that each feature type (color and gradient-based) are useful in their own way, so **combining features** seems to be an interesting thing to do. The simplest way of doing so is to concatenate the feature vector obtained from the color and the gradient-based processings. However, this will provide us with a difference on magnitude regarding the elements of the color part and the gradient part of the combined vector. This is why a **normalization step** may prevent one side from dominating the others when classifying objects. It is also possible to find more elements in one side than in the other. We should try to avoid this, for example, by removing the redundancies. For instance, a **decision tree** could be used to analyze the importance of each feature in order to drop the less relevant ones.

The `06_norm_shuffle.py` script uses the `StandardScaler()` function from the `sklearn` package from Python to normalize data. It requires the data to be a NumPy array where each row is a feature vector. In order to do this, a multi-dimensional list is created and converted to the required format. The `extract_features()` function of this script takes in a list of image filenames, reads them one by one, then applies a color conversion (if necessary) and uses `bin_spatial()` and `color_hist()` to generate feature vectors. The function then concatenates those two feature vectors and append the result to a list. After cycling through all the images, it returns the list of feature vectors.



## Classifier Building
Once being able to extract representative features out of images, the next step is to build a classifier. The simplest way to do so is to sample small patches all over each frame and classify them as *cars* or *not-cars*. To manage to do so, it will be necessary to **train** the classifier.

### Data Preparation
In order to train a classifier, data is a must. In particular, **labeled data** will be the used one. Therefore, samples of both car and non-car images will be needed. In order to do so, it is possible to crop regions of a larger frame and rescale them to a fixed size (the classifier's input size). Moreover, the used dataset should be balanced. i.e. the amount of samples of each class should be similar to the amount of samples of the other. If the difference between the number of samples of each class is too big, the classifier may try to classify almost everything as the class with more samples because during its training, it learnt that, given an input image, it will be more likely for it to be from this class. Techniques such as data augmentation can be useful to balance the dataset.

Once having a balanced dataset, the next step is to split it in a training set and a test set. The testing must be performed using unseen data to get proper measurements on the classifier's performance. It is also convenient to shuffle the data while creating the sets to avoid any kind of dependency in the order it is provided. Again, the sets should be balanced regarding the amount of samples of each class.

### Training
The training phase consist in taking samples from the training set, computing its features and forwarding them through the training algorithm alongside with the labels. This training algorithm will initialize a model and automatically tweak its parameters given the obtained classifications and the actual label assigned to each set of features. In order to do so, an error between the prediction and the ground-truth label is used. The training process will end according to several possible termination criterias, such as when this error is under a certain threshold, when a certain amount of iterations has been completed, etc.

Once the training is over, the test set is used to check how the obtained model performs when evaluating unseen samples. Usually, the test error will be higher than the training one, but both often decrease the more the model is trained, until *overfitting* occurs, which will happen if the model is trained too much. This will make the model to *learn* the training set and not be able to generalize properly to new samples, which will cause the testing error to increase.

Regarding the model, there are many posibilities. In this lesson, a **Support Vector Machine** (SVM) will be used, but some other options are *Decision Trees*, *Neural Networks*, etc. It could even combine multiple classifiers.

### Parameter Tuning
In order to tune the parameters of the SVM that is going to be used, a kernel and a gamma and C paramaters need to be chosen. When using a linear kernel, only C can be tuned. For non-linear kernels, C and gamma can be tuned.

Python's `Scikit-learn` package provides with some hyperparameter tuning algorithms, such as [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) or [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV).

`GridSearchCV` will try combinations of a given list os possible values for each hyperparameter and test them against cross-validation to evaluate which performs better. `RandomizedSearchCV` works similarly, but it uses random combinations of the parameters, which makes it faster because there are some combinations that are not tested.

`GridSearchCV` uses 3-fold cross validation, which consists in dividing the training set into 3 equal parts, use two for training and one for validating. Then, another part is chosen for the validation and the other two are used for training. Finally, it repeats again with the last part. It averages the achieved training accuracies for each partition. This will provide with an accuracy for each tested parameter combination. The combination producing the highest accuracy is the chosen one. A sample usage of this wuold be:

```
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]} # Dictionary of parameters to combine
svr = svm.SVC()                                        # Load the classifier algorithm
clf = grid_search.GridSearchCV(svr, parameters)        # Create the classifier using the algorithm and the parameters
clf.fit(iris.data, iris.target)                        # Try all parameter combinations and return a fitted model with the best performing parameters
clf.best_params_                                       # Access the values of the parameters
```

### Color and HOG-Based Classifiers
In order to try a **color-based classifier**, it is possible to reutilize the `bin_spatial()`, `color_hist()` and `extract_features()` functions from the challenges implemented before in this lesson. With the code in `08_color_classifier.py`, an accuracy of 97.7% is achieved. A way of modifying this result is tweaking the `spatial` and `histbin` parameters.

A **HOG-Based Classifier** is implemented in `09_hog_classifier.py`. It returns a 93% of accuracy, which can be modified with the parameters `colorspace`, `orient`, `pix_per_cell`, `cell_per_block`, and `hog_channel`. `hog_channel` can take values of 0, 1, 2, or "ALL", meaning that you extract HOG features from the first, second, third, or all color channels respectively. 



## Sliding Windows
The idea of this is to scan an image searching the target objects. To do so, a sub-region of the image will be defined and moved accross the image and its content will be forwarded through the trained classifier.

In order to compute how many windows will be necessary for a given image, some information is needed, such as how big the image is, how big will the windows be and how much they will overlap their past positions. Given this information, the number of needed windows is:

```
windows_x = 1 + (image_width - window_width)/(window_width * overlap_proportion)
windows_y = 1 + (image_height - window_height)/(window_height * overlap_proportion)
total_windows = windows_x * windows_y 
```

An script drawing the sliding windows is implemented in the `10_sliding_window.py` file. It is straightforward the main problem of this approach, which is the scale. In an image took from a car driving in a highway or in an urban environment, cars from different sizes will appear, while the sliding windows, so far, will only search objects from a certain size (the ones containable in one window). 

### Multi-Scale Windows
To solve this, ***Multi-Scale Windows*** are used. In order to use them, the first thing to do is define minimum and maximum sizes of the windows. i.e. the minimum and maximum sizes that will be considered for the classification. It is also necessary to define how many intermediate scales will be used. This will increase a lot the number of windows to process, which will slow down considerably the algorithm. Some techniques could reduce this computational charge, such as searching only in zones where vehicles are more likely to be, such as the bottom half of the image. It would also be possible to use only one row of each scale because it is fairly easy to estimate where in the image the big and small cars will appear.

A sliding multi-scale window search is implemented in the `11_search_classify.py` script, which uses the helper functions of `11_lesson_functions.py`. The classifier can be trained using the dataset in the `dataset` folder. Using 9 orientations 8 pixels per cell and 2 cells per block, it provides a 98.5% of test accuracy. In order to get other results, tweaking the parameters mentioned in the *Color and HOG-Based Classifiers* sub-section can do it. It is also possible to modify the searching area with `y_start_stop` in the `slide_window()` function.

### HOG Sub-Sampling Window Search
A way of optimizing the algorithm is computhing the HOG features only once in a way that can be used by the smallest window size. Then, by sub-sampling the generated HOG feature map, features coming from bigger cells will be usable for bigger windows.

In order to test that, the [SVC model](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Vehicle_Detection_Images/svc_pickle.p) and the [test image](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Vehicle_Detection_Images/test_image.jpg) are available in this directory and can be downloaded from the provided links. The files are *svc_pickle.p* and *test_image.jpg*, respectively. This is implemented in the `12_hog_subsample.py` script, which uses the functions of the `12_lesson_functions.py` file.
- Searching and classifying
  - HOG sub-sampling window classifier


## Solving Overlapping and False Positives
The built system is able to scan an image searching potential detections. However, as can be appreciated running some of the scripts listed in the other sections, **false positives** are a thing in the outputted images. Some of them may be just non-cars classified as cars, but they can also be multiple detections of the same car.

An useful strategy to combine multiple detections and remove false positives is creating **heat maps**. A black image will be generated and *1* will be added to every pixel inside each detection. This way, the zones in which several detections have been triggered will have high scores and the false positives will have very low ones. By thresholding the map (usually in taking into account more than one frame), it is easy to say where the cars are and how big (i.e. close to the ego car) they are. 

These are the links to download the [bounding boxes](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Vehicle_Detection_Images/bbox_items/bbox_pickle.p) and the [test image](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Vehicle_Detection_Images/bbox_items/test_image.jpg), even they are provided in this folder as *bbox_pickle.p* and *test_image.jpg*. This technique is implemented in the `13_heat_map.py` script.


## Summary and further considerations
The whole pipeline starts by exploring patches of the image using multi-scale sliding windows. For each of these windows, color and HOG features will be computed and combined in a feature vector that will be normalized. Then, a classifier such a SVM will be used (after being trained) to guess if each of these feature vectors corresponds or not to a car. The detections will be stored and used to generate heat maps that will be thresholded (and compared through time) to provide the final output.

Comparing this traditional approach with deep neural networks, the feature selection (*feature engineering*), classifier training and image search is performed all at once in the neural network, which will work as a black box in which we will not really know what is happening. However, after studying how to solve the problem by hand, we will have an idea of what is likely to be happening inside the networks trained to solve this problem.

