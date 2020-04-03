# Keras

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Keras is an interface that uses TensorFlow, but allows us to implement our designs with less lines of code (sacrifying some flexibility, though). At *Otto*, company that joined *Uber*, they only use Deep Learning for controlling their self-driving car. This can be called *behaviour cloning* because it clones the behaviour that a human would have. It is also known as *end-to-end learning* because it receives the inputs of the sensors and outputs the signals for the actuators. they belive that DL has the potential to change the autonomous vehicle development.

The robotics approach requires a lot of knowledge about sensors, control, planning, etc, while with the DL one, the networks should learn all this on their own. Also, we can autoimprove the algorithm using the data we gather while driving to retrain the network.


## Neural networks in Keras

In this course, all Keras exercises will be performed in JUPYTER workspaces, using python 3.5, Tensorflow 1.3, and Keras 2.09. The [*keras.models.Sequential*](http://faroit.com/keras-docs/2.0.9/models/sequential/) class is a wrapper for neural network models. 

Regarding the **layers**, we have fully connected, max pooling, activations, etc. We use the `add()` function to add each of them to our model as follows:

```
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

# Create the Sequential model
model = Sequential()

# 1st Layer - Add a flatten layer
model.add(Flatten(input_shape=(32, 32, 3)))

# 2nd Layer - Add a fully connected layer (100 outputs)
model.add(Dense(100))

# 3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

# 4th Layer - Add a fully connected layer (60 outputs)
model.add(Dense(60))

# 5th Layer - Add a ReLU activation layer
model.add(Activation('relu'))
```

The dimensions of every layer is computed automatically after we set the ones from the first one. In the *01_traffic_sign_classifier.ipynb* a model is implemented and trained. In *02_traffic_sign_classifier_conv.ipynb*, it includes a convolutional network. Pooling and dropout are introduced in *03_traffic_sign_classifier_conv_pool.ipynb* and in *04_traffic_sign_classifier_conv_pool_drop.ipynb*. The *05_keras_train_and_test.ipynb* example trains (and evaluates) and tests the model.The [MNIST CNN](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py) example can be used as a more detailed template.

NOTE: In Keras, the dropout probability is the probability to **drop** the weight, not to keep it (as it was in TensorFlow).
