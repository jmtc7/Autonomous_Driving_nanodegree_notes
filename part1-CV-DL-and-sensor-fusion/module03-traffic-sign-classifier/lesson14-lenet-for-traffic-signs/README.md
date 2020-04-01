# LeNet for Traffic Signs

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

During this lesson we will deep in the LeNet network to understand all its details in order to adapt its architecture to be able to use it to classify traffic signs instead of numbers from the MNIST dataset.

We will go trough the solution of the LeNet implementation challenge form the last lesson explaining what is happening in every line.


## LeNet implementation analysis

### LeNet data

We start loading the data (provided by TF) and adding a 2 pixels padding to transform from the original 28x28 images to 32x32, which is the input of LeNet. We could also upscale the images using OpenCV or other image processing tools, but this padding solution is way cheaper and does the job. We also visualize one label and its associated input to make sure that everything looks as we think it does. Finally, we pre-process the data shuffeling it. It is key because the order in which a network receives the training data may affect critically to the obtained model (if it only receives 0s, then 1s, then 2s, etc. by the time it finishes with the dataset, it will work fine with 9s but not with 0s).

### LeNet implementation

We set the epochs of the training. In general, the more, the better result we will get, but the training will be longer. We also set the batch size, which we want to be big to accelerate the training, but is limitted by the available memory on our hardware.

We have to make sure that the dimensions of the outputs of one layer match the input dimensions that the next layer accepts. The network input should match our data dimensions and its output should be a vector with as many elements as output classes we want.

### LeNet training pipeline

We start by creating a TF placeholder both for the inputs (x) and the outputs (y). They should be prepared to admit any kind of batch size, so we will use the *None* value in the appropiate element of their dimensions. We also create the one-hot encoded version of our labels, using vectors with as much elements as classes we have (10).

Regarding the training itsel, we will setup a learning rate (0.001 is usually a good value to start with) and start building the rest of our computation graph. We get the logits from the last layer of the already defined architecture. Then, we use the `tf.nn.softmax_cross_entropy_with_logits()` function to convert these logits to probabilities using a softmax and then compare them to our one-shot encoded labels using cross-entropy. With `tf.reduce_mean()`, we average the the cross-entropy of all the images. `tf.train.AdamOptimizer()` uses the Adam algorithm to minimize the loss function in a similar way to how the Stochastic Gradient Descent (SGD) but more sophisticatedly, so it is usually a good default choice for the optimizer. Finally, we use the `minimize()` function of the optimizer to perform backpropagation to update our network's weights in order to minimize our training loss.

### LeNet evaluation pipeline

This pipeline is for the model evaluation. We start checking if a certain prediction is correct by comparing the outputted logit with the one-shot encoded label. We will then use `tf.reduce_mean()` to average all the accuracies into our result. In order to actually run this evaluation pipeline we use the `evaluate()` function defined in this same block. It takes the evaluation dataset as input, batches it and runs it trough the previously defined evaluation pipeline.

### LeNet training

We have already defined LeNet's architecture, the training pipeline and the evaluation pipeline and execution. Now we are ready to train the model using all these resources. We start creating a TF session and initializing the variables and doing as much iterations of the process as epochs are configured. In every epoch, we shuffle our input data to avoid influence our model with the order of the data. We will them divide the data into batches and run the optimization once per batch. Finally, we will evaluate our predictions before starting the new epoch. Once we finish the training, we save it.

During the training process, we see that the accuracy is high from the beginning. This is due to the power of the LeNet architecture but also because our hyperparameter choice is good. This will not be the case for most of our experiments.

### LeNet testing

We should run the data through the test dataset one time, once we finished the training and fine-tuned it to get our target validation accuracy.



## LeNet for Traffic Signs

After restarting the kernel of the notebook to make sure we have a fresh start, we need to change the block that downloads the MNIST data in order for it do download the traffic sign data (this is already done and downloaded in the traffic sign classifier project workspace). The traffic sign images are already 32x32, so we will not need the code that adds these padding pixels that we used for the MNIST images. The traffic sign dataset does not provide a validation set, so we should use the `split()` function from SKLearn library to use a portion of the training set (only) as validation. A 20% should do the job. We can visualize an image of the dataset, keeping in mind that these are **in color**. We should aswell make sure that the label corresponds to the outputted sign. We can do this by checking the .csv file that matches numerical labels to text descriptions.

Regarding the architecture, we will need to **process inputs with 3 channels, not one**. Regarding the logits, we now should have **43 output classes** instead of the 10 we had before. These changes also affect to the input and output TensorFlow placeholders' dimensions. These changes should be enaugh to be able to use the notebook (and even expect decent results).

Regarding how we can improve it, there are several options, such as:
- Modify the architecture or even only the layer's dimensions.
- Add regularization techniques such as *L2 regularization* or *dropout* to make sure to avoid overfitting.
- Tune hyperparameters.
- Improve data pre-processing using normalization and transforming it to have a 0 mean.
- Augment the training data using rotation, shifts or color modifications.

Some other posibilities are displying the training process in a graph, making automatic model saves, etc.



## Bonus: Visualizing layers

Even it is really hard to know what exactly is going on inside a neural network, we can get an idea of it observing the feature maps that the layers inside output. This way, we will be able to see what catches the network's attention. Maybe it finds specially relevant the sign's boundaries or the contrast in the colors of the inner drawings...

For this purpose, we can use the following code, which receives as arguments the image we want to forward trough the network and the variable name of the layer we want to visualize. 

```
# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# Note: that to get access to tf_activation, the session should be interactive which can be achieved with the following commands.
# sess = tf.InteractiveSession()
# sess.as_default()

# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and    max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```

We can use this to compare the response of certain layers when receiving different inputs, such as one with a new sign vs one whose colors have been degradated by the sun or one deformed by some collision. We can also forward the same image in a trained network and compare certain features with the ones generated by an untrained model.


