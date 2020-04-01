# TensorFlow

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The main development of Neural Networks was done in the 80s and 90s but, due to hardware limitations, the field was abandoned. During the last decade, this field has been succesfully applied to speech recognition (2009), computer vision (2012), machine translation (2014) thanks to the increasing ammount of available data and cheap and fast GPUs.


## Basic concepts

NOTE: The course seems to be using TensorFlow 1. In order to use the compatibility version of TensorFlow 2, we will need to add to our code:

```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

This is the easy way, but here is a [guide for migration](https://www.tensorflow.org/guide/migrate) that will make us able to take advantage of the library improvements.

The following is a sample pre-conversion TF 1 code:

```
in_a = tf.placeholder(dtype=tf.float32, shape=(2))
in_b = tf.placeholder(dtype=tf.float32, shape=(2))

def forward(x):
  with tf.variable_scope("matmul", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", initializer=tf.ones(shape=(2,2)), regularizer=tf.contrib.layers.l2_regularizer(0.04))
    b = tf.get_variable("b", initializer=tf.zeros(shape=(2)))
    return W * x + b

out_a = forward(in_a)
out_b = forward(in_b)

reg_loss=tf.losses.get_regularization_loss(scope="matmul")

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outs = sess.run([out_a, out_b, reg_loss], feed_dict={in_a: [1, 0], in_b: [0, 1]})
```

And this is after the conversion (the variables are local Python objects; the forward function still defines the calculation; the Session.run call is replaced with a call to forward; the optional tf.function decorator can be added for performance; the regularizations are calculated manually, without referring to any global collection; and there are **no sessions or placeholders**):

```
W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def forward(x):
  return W * x + b

out_a = forward([1,0])
out_b = forward([0,1])

regularizer = tf.keras.regularizers.l2(0.04)
reg_loss=regularizer(W)

```

### Tensors

Instead of integer, float or string variables, TensorFlow (TF) uses *tensors*, which are n-dimensional arrays. Some examples of constant tensors (their value never changes) are:

- A = tf.constant(1234) # 0-dimensional int32 tensor
- B = tf.constant([123,456,789]) # 1-dimensional int32 tensor
- C = tf.constant([ [123,456,789], [222,333,444] ]) # 2-dimensional int32 tensor

### Sessions

TF's idea is to build a *computational graph*, a way of visualizing mathematical processes. The *sessions* are environments for running graphs. They will send the data to the local/remote GPUs/CPUs that are being used.

A sample code using sessions is the following:

```
with tf.Session() as sess:
    output = sess.run(hello_constant)
    print(output)
```

It creates a session instance (*sess*) and evaluates the *hello_constant* tensor before printing the result.

### Input

When we want to use non-constant values, we use *tf.placeholder* and *feed_dict*.

**tf.placeholder()** returns a tensor containing data passed to *tf.session.run()*, which allows us to decide which data to use just before the session begins. We do this by using the **feed_dict** parameter of the *tf.session.run()* function. Here is how we set several tenshors using *feed_dict*:

```
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
```

If the value assigned to the tensor does not correspond to its type, we will get an error similar to `ValueError: invalid literal for...`.

### Basic math in TF

This are some examples on how to use basic operators:

```
x = tf.add(5, 2)  # Addition. Result: 7
x = tf.subtract(10, 4) # Substraction. Result: 6
x = tf.multiply(2, 5)  # Multiplication. Result: 10
```
Sometimes we may have problems with the tensor types because the math functions often require them to match, such as *substract*


## Linear functions in TensorFlow
It is relevant to highlight that the final scores provided by neural networks are also known as ***logits***.

For the weights and bias in TF, we need a tensor that can be modified, so the placeholders (constant with unknown value until the session is running) and the constants (known constant) are not valid. For them, we will use the *tf.Variable*, which creates tensors with an initial value that can be modified. We need to initialize these tensors using *tf.global_variables_initializer()* before running the session, just like this:

```
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

We must initialize the *variable* tensors. Usually we use random numbers from a normal distribution, which will avoid starting from the same point in every training. Using a normal distribution prevents one weights to start with much more relevance than the others. We do this with *tf.truncated_normal()*:

```
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

bias = tf.Variable(tf.zeros(n_labels)) #The weights are already preventing the network from being stuck, so we can do bias=0
```

A sample on how to train a network (using only one sample) is implemented in the *02_simple_train.py/ipynb* script/notebook.

When using several samples (normal case), we will need to turn the bias into a bias repeating the bias tensors. However, we can use the **broadcasting** operation used in TF and in Numpy to add arrays of different dimensions (when one of them share dimension with the last dimension of the other), like follows:

```
import numpy as np
t = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
u = np.array([1, 2, 3])
print(t + u)
```

### One-hot encoding

It is create a variable that will be activated only when the probability of the class it represents is the highest. They can be really inefficients when the amount of classes increase, which can be solved using *embeddings*, which will be explained later.

We usually use the one-hot encoding vector as a label for training data and compare it with the NN-outputted probabilities using cross-entropy (Entropy = -sum(labels x log(scores))).

### Multinominal logistic classification

Recap: We have inputs that will be transformed into logits by the NN. These logits will be transformed into probabilities by a softmax layer. Finally, we will compute the cross-entropy using these probabilities and the one-hot encoded labels. This process is called **multinomial logistic classification**.

The **loss** will be the average cross-entropy. It will be small if the classification is doing well.


## Numerical stability

When computing maths digitally, we want to avoid working (and especially mixing) too big or too small values because it can lead to the introduction of big errors. This is why we usually want to put use variables with 0 mean and the same variance. 

When working with **images** with pixels values between 0 and 255, we can just substract 128 and divide by 128. This will make our pixel values to be between -1 and 1.

There are many schemes to initialize weights and bias, but a comon simple one is to randomly sample a gaussian distribution with 0 mean and sigma variance. The larger the sigma, the biggest peaks the network will have in its initial states (will be more opinionated). It is usually preferrable to begin with a smaller sigma to have a more uncertain network.


## Generalization in our models

Models will have a tendency of memorizing the training data, so their performance will decrease a lot when feeded with new data. We avoid this by using three sets of data:

- Training data: For the training itself.
- Validation data: To test how well the model generalizes the learning. These tests are done during the training and their outcome will modify the learning.
- Test data: To get a measure on how well the model performs with new data. It is used after the training just to get trustable measurements of the performance. This set is essential because the validation data is undirectly learned by the model.

Regarding the ammount of training/validation/test data is also relevant. If we use 10 times more data, the weaknesses and strengths of our model are most likely to show. If we use too few data, noise can critically affect the performance measurements. A rule of thumb is that a minimum of 30 samples should change in order to be considered a true change in the performance. e.g. having 3000 samples, you will need a variation of a 1% on the performance measurement to consider it true. i.e. moving from 80% of accuracy to 80.5% is not a relevant change, while changing from 80% to 81% would be.

This is why for most approaches, more than 30000 samples are used for validation. This makes the performance changes significant for the first decimal place, which allows a small enough resultion to perceive small improvements. When it is not possible to gather that much data (which is usually the fastest and best way), **cross-validation** is usually a good enough option.


## Stochastic gradient descent

Computing the loss is very costly, and computing its gradient to apply gradient descent is about 3 times more costly. Moreover, we will need to do it a lot of times for a lot of data to train our models. This is why some other approaches are used, such as **stochastic gradient descent** (SGD).

This alternative computes the loss in a small random subset of the data (1 to 100 samples a time). This will make a (bad) estimation of the gradient, but it will reduce a lot the computational requirements, so we can make many more iterations to reach a minimum. This usually gives us way better results and scales really good with data amounts and model complexity.

For this bad estimations to succeed, the data and start points are even more critical. We definatelly need to make sure to be using 0 mean data with small variance for the (truly) random initializations. Some other tricks are:

- **Momentum**: We can use a moving averange of the computed gradient estimations instead of the raw direction from the current data batch. 
- **Learning rate decay**: A critical factor is to decide how smaller should this steps be regarding traditional gradient descent. It a very complex issue, but something that seems to work well is to make them smaller and smaller as the training advances.

**ADAGRAD** is a modification of SGD that automatically uses momentum and learning rate decay. This makes the learning less sensitive to hyperparameters.


## Mini-batching

This is a technique that divides the data into mini-batches (subsets) that allows us to train a model when our hardware is not able to load all the data in memory. The more data considered at a time, the better, because more samples are considered in the loss calculation and in the whole backpropagation process, but this technique is used when we would be unable to train the model at all (because we can store all of our data in our hardware).

It makes a lot of sense to combine it with SGD. Since we will be suffling the data randomly to build our mini-batches, we will be undirectly performing SGD.

### Mini-batching in TF

Given that sometimes our dataset will not allow us to divide it into equally subsets, we can use TF's placeholder to define unknown or *None* dimensions. If we know that the input images will have *n_input* pixels and that we will be working with *n_classes* classes, but we do not know how many samples we will be feeding into the network in each batch, TF can adjust the memory whenever the *features* and *labels* placeholders are initialised. In order to use this functionality, we will indicate a *None* dimension as follows:

```
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
```


## Epochs

An **epoch** is one complete forward and backward pass trough the whole dataset. We do more than one pass to increase our accuracy without requiring more data. We should adjuts the number of epochs and use it as a termination criteria or just use them as a reference of how well our hyperparameter modifications are performing.


NOTE: A whole example pipeline is implemented using TensorFlow in the *07_lab.ipynb* notebook and its equivalent conventional python script.
