# Scene Understanding

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this lesson, Object Detection, Semantic Segmentation and the Intersection over Union (IoU) metric will be discussed.

## Scene Understanding Overview
### Object Detection
**Object Detection** is fairly easy to solve because the neural network will only have to figure out where the target is and draw a bounding box around it. Some open source state of the art solutions are YOLO and SSD. They are fast and effective for many classes.

However, bounding boxes are not always the most convenient marker. For instance, when trying to detect a road, the same bounding box could represent a straight or a curved one. Moreover, if occulssions occur, the bounding box may contain more information about the object that ocludes the target than about the target itself.

### Semantic Segmentation
**Semantic Segmentation** consists in assigning a label to each pixel. Therefore, each region of an image will be *road*, *car*, *pedestrian*, *sign*, etc. This is also known as *scene understanding* and is very relevant to self-driving cars.

An approach to this is using multiple decoders. For example, one for the segmentation itself and other for depth estimation. This would allow the system not only to know what is in the image, but also (roughly) at which distance it is. However, for this lesson, we will focus in using only one decoder to perform the segmentation.

### Intersection over Union Metric
Regarding how to evaluate the quality of a segmentated image, the **Intersection over Union** metric (IoU) is widely used. It consists on dividing the intersection by the union sets of the ground-truth and the predicted masks of each object. The number of pixels that compose the intersection and the union are the used magnitudes. It is an usual practice to compute this metric for every testing image and then compute the average IoU in order to get a general idea of how well the network performs. In TensorFlow it can be done by using the `metrics.MeanIoU()` function.

This is implemented using two 4x4 matrices and four classes for simplicity in the `01_iou.py` script.


## FCN-8's Architecture
Next, the architecture of this FCN will be analyzed to get a better understanding on this kind of networks and some real-world references.

### Encoder
Its **encoder** is the VGG16 network pretrained with ImageNet for classification. Its fully connected layers are replaced by **1x1** convolutions. As an example of doing this, to substitute this fully connected layer with 2 outputs:

```
output = tf.layers.dense(input, 2)
```

This would be its equivalent 1x1 convolution:

```
output = tf.layers.conv2d(input, 2, 1, strides=(1,1))
```

### Decoder
After downsampling the input image with the encoder and using the 1x1 convolutions (instead of the fully connected layer) to keep the spatian information, the data is forwarded to the **decoder**. The shape of the tensor after the final convolutional transpose layer will be 4-dimensional: (*batch_size*, *original_height*, *original_width*, *num_classes*). A sample transpose convolution in TensorFlow would be:

```
output = tf.layers.conv2d_transpose(input, num_classes, 4, strides=(2, 2))
```

Finally, some **skip connections** will be introduced by combining the outputs of two layers: the current one and another further back in the network (usually a pooling one). This combinations of layers can be done by a simple addition using the `tf.add()` function (after making sure that the sizes are equal) and forwarding the result to the next transpose layer. Thisis usually done with the fourth and third pooling layers as follows:

```
input = tf.add(input, pool_4)
input = tf.layers.conv2d_transpose(input, num_classes, 4, strides=(2, 2))
input = tf.add(input, pool_3)
input = tf.layers.conv2d_transpose(input, num_classes, 16, strides=(8, 8))
```

### Classification and Loss
Once having the network defined, it is time to define a loss function to be minimized during the training. Since this is a classification problem, the already studied **cross entropy** could do the job. However, in order to use it, it is necessary to reshape the 4D output to a 2D one in which each row will be a pixel and each column, a class. This is done as follows:

```
logits = tf.reshape(input, (-1, num_classes))
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
```

### Object Detection Lab
All this can be used to implement a Single Shot Detector (SSD) able to implement relevant objects for self-driving cars, such as other cars or traffic lights. The iPython Notebook provided in the [Object Detection Lab](https://github.com/udacity/CarND-Object-Detection-Lab) can be modified to achieve the desired detections.

To use it locally, it is convinient to create the provided Anaconda environment using the `environment.yaml` file like this: 

```
conda env create -f environment.yml
```

It may be necessary to chose the pip installation of TensorFlow to tensorflow-gpu or just tensorflow depending on if a GPU is available in your system. Other useful **resources** are the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and this [driving video](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/advanced_deep_learning/driving.mp4).


