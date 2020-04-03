# Transfer Learning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Transfer Learning consists in, instead of starting from scratch, start building a networ using a previously trained model and then fine-tune it. This is way quicker than creating an architecture, training it, experimenting with it and fine-tuning it and often can provide even better results. Usually this is done choosing a network that was designed and trained to solve a similar problem to the one we have.

If we choose a serious work that has been done over months that was obtained after days or weeks of training, there is a lot of intelligence stored there that we can take advantage of. Moreover, if our dataset is small, we can not train too much our models, so we can use our limited resources to fine-tune an existing solution.

During this lesson, we will study several options that are widely used for Transfer Leraning.


## Transfer Learning usage - practical cases
We will implement TL in one or other way depending on two main factors:
- How similar is the data of our problem to the data with which te network we are using was trained with.
- How big is our data set (a big one can be one million of images and a small one 2000, is is very relative).

### Case 1: Similar data, small dataset
We will **remove the end of the network and add one new fully connected layer** that gives us as many outputs as we need (number of classes in the new dataset). We will **randomize** the weights of this **new layer and freeze the rest** of them. Finally, we train this last layer to adapt the knowledge to our data.

The weight freezing is to avoid overfitting (given that we have few samples). We are assuming that the high level features of both datasets will be similar.

### Case 2: Different data, small dataset
We will **remove the layers near to the beginning** of the network, the ones that extract the high-level features. We will **add a fully connected** with the outputs that we need and, again, **randomize** its weights **and freeze** the ones of the pre-trained model.

This workflow will use the low-level knowledge of the pre-trained model (lines, edges, shapes), while removing the high level one specific of the dataset with which the network was trained. Again, we freeze the weights to avoid overfitting.

### Case 3: Similar data, large dataset
**Remove the last fully connected and replace it with one that matches our number of classes**. Randomize the weights of the new layer and use the pre-trained weights to initialize the ones of the rest of the network. Finally, we will **retrain all the weights** with our data.

Having a lot of data, overfitting is not a big problem. Since the data will be similar, we can use all the network (including high-level features).

### Case 4: Different data, large dataset
**Remove the last** fully connected and **add another one matching our class number**. Now we can:
- Try to retrain all as in the last case (3, similar&large).
- Retrain with every weight randomized (using only the architecture).



## Deep Learning history
It all started back on the 90s with the Yann LeCun's LeNet network, used by post offices to recognize hand-written numbers. However, due to the lack of data (solved with the Internet) and computational power (solved with the latest advanced in CPUs and, specially, GPUs).

The [ImageNet](http://www.image-net.org/) data set is a huge collection of hand-labelled images. Due to its existance, the ImageNet Large Scale Visual Recognition Competition, an annual competition about building the best networks for object detection and localization. Now it is hosted in [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge) and it provides 50000 images of objects of 1000 different classes. 

This big ammount of data makes it really common to use ImageNet to pre-train networks for general/Transfer Learning purposes. [Keras Applications](https://keras.io/applications/) can be used to import pre-trained networks.

Note: Given the huge ammount of classes in ImageNet, it is highly ulikely that we can consider our data *different* from the one which was used to train the networks explained in the next subsections. i.e. chances are, we will not fall in the *different* data cases explained on the previous section.


### AlexNet
It reused the concepts in LeNet, but used the highest-end GPUs on 2012 to decrease the training time. It was a pioneer in the usage of Rectified Linea Units (ReLUs) and dropout to avoid overfitting. The winner of last year succeeded classifying the 74% of the images and AlexNet raised it to 85%.

In their [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), they explain how parallelizing the network in two different GPUs (communicated in some points of the network) increased the accuracy by 1.7%.

Even today, AlexNet is used as a starting point. Usually, we use a simplified version of it that removes unnecessary features.


### VGGnet
In 2014 two teams were very close to each other with a 93% of accuracy. One of them was using [VGGnet](https://arxiv.org/pdf/1409.1556.pdf) (or VGG). Its architecture is simple, which makes it great for Transfer Learning. It is formed by 3x3 convolutions, 2x2 pooling layers and three fully connected layers. Its strength is flexibility.

Keras provides two version of VGG: (1) VGG16 and (2) VGG19. The numbers are the number of layers. We can use the VGG16 as follows:

```
from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet', include_top=False)
```

We can use *None* instead of *imagenet* to use random weights. We can include or not the last fully connected layer using the *include_top* argument. The input of this network is 224x224.


### GoogLeNet
The network submitted by Google to the ImageNet competition. It performs slightly better than VGG, but its main advantage is how fast it runs (almost as AlexNet). Its secret are the *inception modules*, which, as explained before, consist in concatenating the output of an average pooling followed by a 1x1 convolution and a 1x1, 3x3 and 5x5 convolutions as the output of the layer (all of them will be applied to the same feature map, they will not be sequential). This can be configures to have a small number of parameters, that is what gives the speed to GoogLeNet. Its paper can be accessed [here](https://arxiv.org/pdf/1409.4842.pdf).

To use this architecture in Keras, we will do:

```
from keras.applications.inception_v3 import InceptionV3

model = InceptionV3(weights='imagenet', include_top=False)
```

Note: The original inception model input was 224x224 (as the one from VGG) but the one of InceptionV3 is 299x299.


### ResNet
The 2015 winner was [ResNet](https://arxiv.org/pdf/1512.03385.pdf), the Microsoft Research's submission. Its main difference is that it has 152 layers (AlexNet had 18, VGG 19 and GoogLeNet 22). It is mainly as repeated VGGs, but with connections that skip layers in order to be able to train such a deep network. It archived a 97% of accuracy, outperforming the human accuracy.

We can use it in Keras by:

```
from keras.applications.resnet50 import ResNet50

model = ResNet50(weights='imagenet', include_top=False)
```

NOTE: ResNet50 uses a 224x224 input.

