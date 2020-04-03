# **Traffic Sign Classifier** 
### Project from the third module of the Self-Driving Car Engineer Udacity's Nanodegree

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The aim of this project is to be able to classify traffic signs that appear in 32x32x3 RGB images that can be from 43 different classes using Deep Learning. In order to do so, I will be using the data from the [**German Traffic Sign Benchmark**](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) and will be basing my approach in the **LeNet-5** architecture. The quantified target is to go over a **93% of validation accuracy**. 

In particular, the goals of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with this written report

As the main results, the archived **validation accuracy** during training was **99.2%** and the **test** one is **94.4%**.

[//]: # (Image References)

[image1]: ./readme_images/01_data_overview.png "Dataset visualization"
[image2]: ./readme_images/02_class_samples.png "Samples of each class (original)"
[image3]: ./readme_images/03_class_samples_augmented.png "Samples of each class (after data augmentation)"

[image4]: ./readme_images/04_lenet_original.png "Original LeNet-5 architecture"
[image5]: ./readme_images/05_lenet_mod.jpeg "Modified LeNet-5 architecture"
[image6]: ./readme_images/06_accuracy_vs_epochs.png "Validation accuracy evolution"

[image7]: ./readme_images/07_new_images.png "New images collection"


## Data set summary, exploration and preparation
### Data set summary
The used dataset is the [**German Traffic Sign Benchmark**](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news), which contains more than 50000 lifelike 32x32 RGB images of more than 40 classes of traffic signs. Each traffic sign is used only once in the dataset and some difficulties are present, such as blurred images, images taken with too much or too few light, etc.

### Exploratory visualization
Here is an exploratory visualization of the data set. Some examples from the dataset can be seen.

![Data set examples][image1]

Next, a bar chart showing how many samples of each class has the training set is shown.

![Training samples distribution][image2]

It can be appreciated the big different between the ammount of samples of each class, which may difficult the classification of the classes with less examples (not now because I augmented the data to have more samples of the classes with lower representation). This is why I deceided to perform **data augmentation**.

### Data pre-processing
Given that the shape is the most relevant feature of the traffic signs (frame and drawings inside), I converted to grayscale. Next, in order to have a mean closer to zero, I normalized the images 

List the used techniques and the reasons why they were chosen.

### Data augmentation

Given the huge lack of homogeneiety in the amount os samples of each class (some have more than 2000 examples and others just over 200), I will perform a data augmentation based on the one on [this](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) repository that will generate samples for the clasess represented by less than 750 examples until they have at least this ammount of samples.

This was performed by applying 4 random transformation (translation, scaling, warping and brightness) to an original image of the class that is being augmented in order to generate a new sample. The samples of each class after the aumentation is shown next:

![Training samples distribution after data augmentation][image3]



## Architecture design and testing
### Model architecture
Even I started doing some trials with the original LeNet-5 architecture, I ended up using the modified version of the LeNet-5 network proposed in the [Sermanet/LeCunn's approach](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). They add an **extra convolutional layer in parallel** to the flattening of the last pooling one that will be flattened and concatenated with the output of the other flattening. This gives an array of 800 elements that will be compressed into a 43-element representation by **only one fully connected** layer. These 43 elements will be the logits of the network.

The next image is the original LeNet-5 network:

![Original LeNet-5 architecture][image4]

And this one is the modified version:

![Modified (and chosen) version of LeNet-5][image5]

Note: Both images have been extracted from [this](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) repository.

In particular, the used model starts receiving  32x32x1 images that are forwarded in a convolutional layer that will output 28x28x6 data. This will be sent to a max pooling layer that will transform it into 14x14x6. Next, the second convolutional layer will use this to generate a 10x10x16 representation of the data that will become 5x5x16 thanks to the second max pooling layer. Now comes the **first network modification**. The **network bifucates** into two paths, one that just flattens this information into a 1x1x400 vector and anotherone that uses a convolution to get to this shape. Then, both **vectors are concatenated to go back to a single-threaded network**, getting a 1x1x800 elements data vector. Finally, and this is the **second network modification**, this is transformed to the outputted 1x1x43 logits by **only one fully connected layer**. All of the layers use a stride of 1 in X and Y and no padding (except the pooling ones that use a 2x2 stride).

I also use **dropout** with a keeping probability of 75% to foment redundant learning to increase the network's robustness. 


### Model training
I started training with very standard settings (30 epochs, 128 batch size and 0.001 learning rate). As soon as I did the improvements to the model (mainly, data augmentation and network modifications), I realized that **I needed more epochs** to let my model learn more, so I increased them to 50, then to 60 and finally to **75**.

I noticed that the variations in the accuracy curve were pretty abrupt, so I tried decreasing the **learning rate** to **0.0009** to archive a smoother progress.

The last observation is part of the reason why I decreased the **batch size** to **100**. I wanted a slower training. Moreover, I thought that it may cause a similar effect to the one caused by the Stochastic Gradient Descent (SGD).


### Results
After all my work, I managed to get a **99.2% of validation accuracy**, which decreased to **94.3% when testing** it with absolutelly new images from the dataset that were not related at all with the training or validation sets (both their original and augmentated versions). The next is an image showing the reached validation accuracy and its evolution across the training process:

![Validation accuracy evolution][image6]

The testing accuracy can be seen (or re-computed) using the iPython notebook containing my approach.



## Model testing with new images
### New image aquisition
For this task, I surfed the internet looking for pictures of traffic signs that may be specially challenging in order to test this project. These are the chosen ones:

![Collection of new images][image7]

Most of them are not a perfect square almost fitted to the sign, which is already a change from the dataset images. In particular, these are the reasons why I chose each of them:
- **30 dm/h maximum speed**: It has watermarks making the sign discontinuous. 
- **Right of way**: It is old, dirty and scratched.
- **Road works**: It is scratched, the sun is being reflected on it and the background is unusual.
- **Stop**: This was just a normal, ideal(ish) case to see if something unexpected happen.
- **General caution**: I wanted to test my model using a drawing. It might be exposed to similar data if used in a simulation.

### Performance on new images
As can be see nin the iPython notebook, the model classifies properly 4 of the 5 images and in the misclassified one, the correct answer is the 2nd option of the network (seen on the top probabilities comparison).

