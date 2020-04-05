# **End-to-end CNN for self-driving cars | Behavioral cloning** 
### Project from the fourth module of the Self-Driving Car Engineer Udacity's Nanodegree

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The aim of this project is to train a neural network using Transfer Learning so that it will be able to drive a car in the [Udacity CarND Simulator](https://github.com/udacity/self-driving-car-sim) using only monocular visual information. The simulator allows to manually drive the car to get thrusting, braking and steering data linked to visual information  captured by three cameras located on the front of the vehicle.

In particular, the goals of this project are the following:
- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

The outcome of this project was an end-to-end CNN capable of driving smoothly and at maximum speed without causing any dangerous situations, as prooved in the following YouTube demo:

This work will be followed by a brief documentation/overview contained in this file. This project is a completed version of the sample project template provided by the Self-Driving Car Engineer Udemy's nanodegree. The un-completed original version is [this repository](https://github.com/udacity/CarND-Advanced-Lane-Lines).

A demo video of the performance of this approach is available in YouTube:
[![Demo video](https://img.youtube.com/vi/UwDVvA04Cvg/0.jpg)](https://www.youtube.com/watch?v=UwDVvA04Cvg)


[//]: # (Image References)

[image0]: ./readme_images/00_data_samples.png "Samples of the data"
[image1]: ./readme_images/01_original_6459samples_86classes.png "Original samples per label"
[image2]: ./readme_images/02_combined_19377samples_258classes.png "Multi-view augmented samples per label"
[image3]: ./readme_images/03_nvidia_architecture.png "Original network architecture"
[image4]: ./readme_images/04_training_10ep_combinedData.png "Epochs of the training"



## Files and running instructions
My project includes the following files:
- ***model.py*** containing the script to load and augment the data and create and train the model. In the current version, however, a *generator* is not used because I could develop everything without it with my own hardware, but it is one of the improvements that is planned to be added in the near future.
- ***drive.py*** for driving the car in autonomous mode.
- ***model.h5*** containing a trained convolution neural network. 

Using the [autonomous car simulator](https://github.com/udacity/self-driving-car-sim) provided by Udacity in *autonomous mode* and my *drive.py* file, the car can be driven autonomously around the track by executing:
```sh
python drive.py model.h5
```

The *model.py* is full of comments explaining the implementation details, so this file will be focused in the explanation of the general ideas and the higher level concepts of the project.



## Model architecture and training strategy
### Model architecture
I started experimenting performing Transfer Learning (TL) with LeNet-5 and, later on, with VGG-16 and GoogLeNet (InceptionV3) but I ended up using the architecture proposed by NVIDIA in [this paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). It consists on a normalization layer followed by 5 convolutional layers and 3 fully connected ones, as can be appreciated in the following diagram (from the paper):

![Backbone architecture (before modification)][image3]

**Before the normalization**, I added a cropping layer (both of them using Lambda layers) to remove some useless information from the top and bottom of the image.

Regarding the **convolutional** layers, they are the following:
- 24 ReLU-activated filters with a 5x5 kernel and 2x2 stride
- 36 ReLU-activated filters with a 5x5 kernel and 2x2 stride
- 48 ReLU-activated filters with a 5x5 kernel and 2x2 stride
- 64 ReLU-activated filters with a 3x3 kernel and 1x1 stride
- 64 ReLU-activated filters with a 3x3 kernel and 1x1 stride

Finally, the **fully connected** layers are 100, 50, 10 and 1 neuron, which will be the one outputting the predicted steering angle.

### Overfitting reduction
The main approach I followed to reduce overfitting was the usage of a lot of **data**. After several trials with different source data sets, I ended up using 3 counter-clockwise laps (one of them beginning and ending in another point of the circuit) and one clockwise to increase the car's hability in turning to the right.

Moreover, I used the technique proposed [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by NVIDIA to increase the recovery performance and introduce more turning samples to prevent the model to try to go straight all the time. It consists in using **three cameras** in the front of the car, one in the center and other two moved to the sides of the car. The registered steering angle is associated as the label for the image from the camera on the center and an empiric-obtained correction is applied to it to get the labels for the images on the left and on the right.

In future versions of this project, I would like to make more in-depth tests to detect if the car is overfitted to the tested circuit and, in this case, introduce more measures such as dropout.

### Parameter tuning
I did not have much tuning to do. I mainly tested with changes on the architecture and the data. However, the next are the parameters that could be used to tune the network performance:
- **Epochs**: I tried different ammounts of epochs while using different architectures and ammounts of data. I started with 3 epochs but I had to increase them up to 10 as I was starting to use more and more data and complex architectures.
- **Steering correction** for multi-view data augmentation: I tried 0.2 based on the turns I was doing while driving to generate data and it seemed to perform well, so I did not experiment with more values.
- **Test/validation split**: I have used the general 80%/20% split. Again, I did not change it because the performance was very nice.


### Training data
In addition to the details given before in the *Overfitting reduction* sub-section, I tried to keep the speed around **20-25 MPH** to manage to archive a smooth driving without being challenged by driving at high speed (as the resulting network is capable of doing ;D), keeping the car in the **center of the lane**, even during curves.



## Architecture and training documentation
I start by **loading the data** generated by me using the simulator. The loading is done by reading the CSV file containing the paths to the center, left and right images and the steering angle that was being inputted for each frame. As explained, I perform the viewpoint-augmentation and I apply the corrections to the steering angle in order to match it with the left and right images.


Next, I show some **insights on the availabe data**. For reference, without the *data augmentation* with left and right images, i had over 6500 images and only 86 unique steering angles. After the viewpoint-based corrections and augmentation, I archived more than 19000 images and more than 250 steering angles. The samples for each steering angle can be seen in the next images:

![Original samples per label][image1]

![Viewpoint-augmented samples per label][image2]

Obviously, the second graph corresponds to the augmented samples per label. It can be clearly appreciated that:
- A lot more of samples per label are available
- It has the shape of a 0-centered gaussian distribution with a lot of samples for a lot of different turns, specially the ones at short/medium distance from the mean.
- In general, we have more labels and more samples for each of them.

I also show some sample frames captured by the car and the steering angle vinculated to them:

![Samples of frames and steering angles][image0]


The next step is to define the network architecture, which is based in the end-to-end network proposed by NVIDIA for self-driving cars in [this paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The main differences are that I added an extra lambda layer to remove useless information from the pictures and that I use RGB instead of YUV.

It is relevant to mention that the fully connected layers from the last part of the network have no activation and converge into only one neuron because this is a regression problem to get the steering angle, not a classification one such as in the traffic sign classifier project.


The last step is to actually **create the model and train** it. Again, I used the Mean Square Error (MSE) instead of cross-entropy because I was solving a regression problem, not a classification one. The chosen optimizer is ADAM (I have not experimented with any other one because this gave proper results and learning). Finally, as metnioned before, I train the model making a train/validation split of 80/20%.
