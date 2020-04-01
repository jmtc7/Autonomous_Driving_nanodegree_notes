# Convolutional Neural Networks

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

They were invented over 30 years ago, but since 2012 they have become really popular in the Computer Vision field. 

The *translation invariance* tells our networks that it has to, for example, find dogs, wherever they are on the images. If it knows that dogs in the bottom left corner and dogs in the top right one will look the same, it will be easier for the model to adjust itself to solve the problem. Same happends with text processing. The word *dog* means the same wether it is on the first or in the fifth line. In these situations, we can use **weight sharing** to help the network learn what a *dog* is in order to be able to use this knowledge in several parts of its architecture. This consists in, when we know that two different inputs contain the same information, we will share their weights. This idea will lead us to Convolutional Neural Networks (CNNs) for images or to Embeddings and Recurrent Neural Networks for text and other types of sequences.

## Introduction to CNNs

The CNNs or *covnets* share their parameters across the space (the convolution kernel). We will substitue the weights for blocks of convolutions. They will reduce the height and width of our image and increase its depth (channel number). The more depth we get, the more semantic complexity we will be able to model. At the end we will have a classifier that will turn really high-level features into our output (i.e. which animal apears in the picture).

Some essential concepts are:
- **Kernel**: The area of the image which is being used with *shared weights* to compute an element of the layer's output.
- **Stride**: The ammount of pixels the convolution moves in each step to process the input. A stride=1 makes the output to have the same size of the input, while a stride of 2 will half the original size (roughly in both cases, it is affected by the padding).
- **Padding**: Add 0s to the edge of the image to keep the original dimensions of the image in the layer's output. When we add a frame of 0s is known as *same padding*. If we add no padding, the dimensions of the output image will be the ones of the input -2 pixels in X and Y and it will be called *valid padding*.

What CNNs do is to start extracting basic features from the image (such as edges) and start combining them to get more and more complex structures, such as shapes, noses, eyes, tires, etc. that they will be able to use to tell if the input image is from a dog or from a car.

The ammount of filters/convolutions in a covnet is known as *filter depth*. It is important to have several of them because, while one may focus in searching for a certain shape, anotherone may focus in an specific color. Each layer will have *K* filters and each of them will add one dimension to the output's depth (number of channels).

Wrap up: To implement a CNN, we need to choose our convolutions, stack them, reduce the width and height of the information in each layer using the stride while increasing the depth using more and more filters. Finally, once we have a deep narrow description of the image, we can process it with a few conventional fully-connected layers and forward the obtained logits to a softmax to get the percentages that will be compared with the one-hot encoded labels.

Regarding the output dimensions of a convolutional layer, we can follow these formulas:

```
new_height = (input_height - filter_height + 2 * Padding)/Stride + 1
new_width = (input_width - filter_width + 2 * Padding)/Stride + 1
new_depth = n_filters
```

In TF, a layer that receives a 32x32x3 input and uses 20 8x8x3 filters and a stride of 2 and a padding of 1, will be implemented as follows:

```
input = tf.placeholder(tf.float32, (None, 32, 32, 3)) # We do not know how many images will be in the minibatch, so "None" is used
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) #(height, width, input_depth, output_depth or number of filters)
filter_bias = tf.Variable(tf.zeros(20)) # One bias per filter

strides = [1, 2, 2, 1] # (batch, height, width, depth)
padding = 'SAME' # same pading is the same as 1 pixel padding # same pading is the same as 1 pixel padding

conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
```


### Basic CNNs in TensorFlow

TF has the `tf.nn.conv2d()` and `tf.nn.bias_add()` functions that we can use to create convolutional layers as shown in the next code. We need the *bias_add*-specific function because the conventional `tf.add` will not work if the arguments do not have the same dimensions.

```
# Output depth
k_output = 64

# Image Properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])

# Weight and bias
weight = tf.Variable(tf.truncated_normal([filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)
```

## Advanced covnet-ology

### Pooling

We can use **pooling** to reduce the dimensionality of our feature maps. We can also use the stride for that, but we lose information. *Pooling* consists in combining several of the obtained results. The most common pooling operation is ***max pooling***, which consists in taking a neighbourhood and keeping only the highest value on it. This does not increase the number of parameters and usually provide more accurate results. However, since we are substituting stride with this, the computation cost is higher and it also introduces the pooling kernel size and its stride.

A common architecture of CNNs is alternate convolutions with max pooling several times and end up with a few fully-connected layers that will send their logits to a softmax.

Another frequent pooling is the *average pooling*, which preserves the average value of the neighbourhood.

In TF, we can use the `tf.nn.max_pool() ` function as follows:

```
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
conv_layer = tf.nn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)
# Apply Max Pooling
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
```

### 1x1 convolutions

Instead of looking to a patch of the image, they only look into one pixel. This means that, instead of a linear classifier for the whole path, if we add a 1x1 convolution after it, we will have a mini NN running over the patch. This allows us to make our models deeper and with more parameters in a cheap way and without changing the CNN's structure. They are cheap because they are not convolutions, only matrix multiplications.

### Inception modules

They use both pooling and 1x1 convolutions. The idea is to be able to choose at every step what is the model going to apply (pooling or convolution? If it is a convolutionm, of which size?) This is done by, in one layer, introduce an average pooling followed by a 1x1 convolution, then a 1x1 convolution, then a 1x1 conv followed by a 3x3 one and then a 1x1 followed by a 5x5. Finally, we will just concatenate the 4 outputs.


## Scripts

In the scripts contained in this folder, you will be able to find a TensorFlow1 implementation of a convolutional layer (*01_tf_conv_layer.ipynb*), another of a pooling layer (*02_tf_pool_layer.ipynb*) and anotherone of the LeNet network (*03_LeNet_Lab.ipynb*)





