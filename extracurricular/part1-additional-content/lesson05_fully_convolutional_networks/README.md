# Fully Convolutional Networks

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this lesson, Fully Convolutional Networks applied over Object Detection and Semantic Segmentation, and Inference Optimization.


## FCNs Overview
A typical CNN is composed by a series of convolutional layers followed by some fully connected ones that will end with a softmax layer if the problem that is being solved is a classification problem (i.e. the output will be probabilities of belonging to each possible class).

However, if the problem to be solved is a detection problem (i.e. know where in a picture there is an object of a certain class), the fully connected layers will not conserve the spatial information. This is when **Fully Convolutional Networks (FCNs)** come into play. Substituting the fully conected layers with more convolutional ones that upscale back the image, it will be possible to **keep this spatial information**. Another advantage of that is that, since convolutional layers do not care about the input size, they should work with **images of any size**.

The **techniques** that have made FCNs achieve state-of-the-art results are:

- Replacing fully connected layers by **1x1 convolutions**.
- Upsampling using **transposed convolutions**.
- **Skip Connections**.

The **architecture** of the FCNs is usually composed by two parts: a *encoder*, which reduces the size of the image keeping the most relevant information of it, and a *decoder* which upscales the image back to its original size, generating one label for each pixel given the features extracted by the encoder.

### 1x1 Convolutions
In order to substitute the fully connected layers by layes with 1x1 convolutions, the number of kernels of the new layer will be equal to the number of outputs in the fully connected layer that is being substituted. The script `01_1x1_convolutions.py` creates two networks, one with a fully connected layer and another one with a convolutional one using 1x1 kernels. It demonstrates that the underlying math is the same (they provide the same outputs).

### Transposed Convolutions
Also known as *deconvolutions*. They are reverse convolutions in which the forward and backward phases are swapped. This [upsamples](https://en.wikipedia.org/wiki/Upsampling) the previous layer. This is usually combined with [interpolation](https://dspguru.com/dsp/faqs/multirate/interpolation/), but the convolutions will learn how to upsample during the training. Some convolution-related animations can be found [here](https://github.com/vdumoulin/conv_arithmetic).

As in conventional convolutions, the stride, padding and kernel size determines the output size. In the `02_transpose_convolutions.py` script, a transpose convolution-based layer is used in TensorFlow.

### Skip Connections
They are a way of retaining the information lost during the encoding. They consist in connecting the output of a layer to the input of another non-adyacent one. This can be done, for example, using an element-wise addition.

An example can be seen in the FCN-8 architecture, which contains two skip connections vs the FCN-32, which has no skip connections and achieve worse results.


## FCNs in Practice
Using features from the encoder and using them with another network (the decoder) is similar to transfer learning. Some techniques of this methodology can be used for Encoder-Decoder (or FCNs) networks. It is common to use encoders pretrained with ImageNet.

Then, the 1st FCN technique is applied (the 1x1 convolutions). After this, the transposed convolutions (decoder) are added. Finally, some skip connections are introduced. It is important to avoid using too many of these because they can increase the model size a lot.

As an example, when using VGG-16 as encoder, only the third and fourth pooling layers are usually used for skip connections.

