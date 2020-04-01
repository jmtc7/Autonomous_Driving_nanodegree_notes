# Deep Neural Networks

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

We will deep into concepts related to neural networks such as activation functions, normalization, regularization and dropouts. During the first part of the lesson we will convert our image classifier from the last lesson into a deep neural network. In the second part, we will deep into how the optimizer does all the hard work computing the gradients. Finally, we will have a look on *regularization*, which is a thing that allows us to train much larger models.

Regarding our last model, we have image inputs of 28x28 (a total of 784 inputs) and 10 output classes, which means 10 neuron. This means that we will have 7840 weights (784 weights per neuron) and 10 biases, which are 7850 parameters to be adjusted in our network.

Even we have a decent amount of parameters, they are organized in a linear model, which means that if two inputs interact in a non-linear way (such as multiplicatinos, sinusoidals, exponentials, etc.), our model will not be able to model this interaction properly. However, using linear operations have advantages such as efficiency, availability of hardware specifically design for them (GPUs) or the stability on the magnitude in the output and outpus. In order to keep these advantages while having an overall non-linear model, we introduce non-linearities.


## REctified Linear Unit (ReLU) activation function

It is y=x if x>0 and 0 if x<0 (*f(x) = max(0, x)*). They also have a nice derivative (0 for x<0 and 1 for x>0). We can take our linear classifier and use two weights and bias matrices, the firsts will connect the input to the ReLUs and the seconds will connect the ReLUs' output to the output of our model (which will provide us with logits that we will forward to a softmax to get the probabilities that we will use to compare with our one-shot encoded labels). This way, we keep the linear operations for the parameters but we have also introduced non-linearity on the model with the ReLUs. We will also have another parameter, the number of ReLUs (H).

The number of parameters now will be *relu_layer_weights + relu_layer_biases + output_layer_weights + output_layer_biases = n_inputs x n_relus + n_relus + n_relus x n_outputs + n_outputs*. What we have done here is adding an extra layer that uses ReLU as its activation function and will be the input layer of our network. It will forward its outputs to a conventional layer (than can use the conventional sigmoid activation function), which will be the outputting the logits that the softmax will transform in the outputted probabilities of the model.

### ReLU in TensorFlow

It is implemented as `tf.nn.relu()` and we can introduce hidden ReLU-activated layers as follows:

```
hidden_layer = tf.add(tf.matmul(features, hidden_weights), hidden_biases) # Y_h = W1*X + b1
hidden_layer = tf.nn.relu(hidden_layer) # Y_h = ReLU(Y_h)

output = tf.add(tf.matmul(hidden_layer, output_weights), output_biases) # Y_final = W2*Y_h + b2
```

NOTE: Fully implemented in the *01_tf_relu.ipynb* notebook.


## Chain rule and back-propagation

It allows us to derivate a complex function knowing the simple derivatives of the functions that compose it. In backpropagation, after propagating the input forward, we backpropagate the output using the chain rule, producing gradients that are used in combination with the learning rate to correct the parameters. This derivatives are usually automatically computed by the DL frameworks such as TF, PyTorch or Keras.

It is important to keep in mind that the back propagation uses aproximatelly two times the memory used for the equivalent step during forward propagation. This is esential for chosing the size of the model because it will be conditionated by your hardware.


## Performance improvement

### Width vs depth

Adding more layers usually works better than making them bigger. This makes sense because most natural processes start using simple information and combining it to get more and more abstract knowledge (e.g. such as lines that get combined to make shapes that get combined to make animals).

### Saving models

After a training that has been running for hours or days, if we close our TF session, all the obtained information will be lost. That is why TF has the `tf.train.Saver` class to save any `tf.Variable` using `saver.save(sess, './model.ckpt')`. We can restore them in our codes using `saver.restore(sess, './model.ckpt')`.

We can build a model:

```
# Remove previous Tensors and Operations
tf.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('.', one_hot=True)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

Train it and save the result:

```
import math

save_file = './train_model.ckpt'
batch_size = 128
n_epochs = 100

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(n_epochs):
        total_batch = math.ceil(mnist.train.num_examples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)
            sess.run(
                optimizer,
                feed_dict={features: batch_features, labels: batch_labels})

        # Print status for every 10 epochs
        if epoch % 10 == 0:
            valid_accuracy = sess.run(
                accuracy,
                feed_dict={
                    features: mnist.validation.images,
                    labels: mnist.validation.labels})
            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                valid_accuracy))

    # Save the model
    saver.save(sess, save_file)
    print('Trained Model Saved.')
```

And, finally, reloading the model:

```
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))
```

However, if we want to **finetune** a previously created model, we may find some problems due to the automamtic *name* asignement that TensorFlow performs. In order to solve them, we have to manually assign the names we want for the weights and the biases (otherwise, when loading the saved data, we might assign weights to biases and biases to weights). This is done as follows:

```
import tensorflow as tf

tf.reset_default_graph()

save_file = 'model.ckpt'

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')

saver = tf.train.Saver()

# Print the name of Weights and Bias
print('Save Weights: {}'.format(weights.name))
print('Save Bias: {}'.format(bias.name))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_file)

# Remove the previous weights and bias
tf.reset_default_graph()

# Two Variables: weights and bias
bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')
weights = tf.Variable(tf.truncated_normal([2, 3]) ,name='weights_0')

saver = tf.train.Saver()

# Print the name of Weights and Bias
print('Load Weights: {}'.format(weights.name))
print('Load Bias: {}'.format(bias.name))

with tf.Session() as sess:
    # Load the weights and bias - No Error
    saver.restore(sess, save_file)

print('Loaded Weights and Bias successfully.')
```


## Regularization

We were not using T-models before because we did not realized about their performance because it is revealed when using enough training data, which was not allways available for the research teams. Also, now we know how to train bigger models using better regularization techniques. Usually, the best possible network for our problem/data is very hard to optimize. This is why we usually use networks that are slightly bigger than what we really need and try to avoid overfitting.

A way of avoid overfitted networks is to look at the accuracy archived with the validation set. As soon as it stops improving, the network is starting to be overfitted to the training data and losing generalization capabilities. **Regularization** is another way of avoiding overfitting, which is to apply artificial constraints to the network implicitly reducing the number of free parameters while not making it harder to optimize. Example of this are **L2 regularization** and **dropout**.

### L2 regularization

It consists in adding a new term to our loss for penalizing high weights, which is usually done by adding the L2 norm of the network's weights to the loss multiplied by a small constant (beta). The L2 norm is the sum of the squares of the individual elements of a vector.


### Dropout

It is another regularization technique. It consist in taking the output of one layer that goes as an input of the next layer (activations) and, randomly, for every sample of the train data samples, we will set half of them to 0. This is equivalent to distroy a random half ammount of the flowing data, which forces the network to learn a redundant representation of the information because it can not be sure that any input will get to any point. This way, even if some activation is removed, the network will learn to have others that will provide the same information. this makes the networks more robuts and less prone to overfitting.

If dropout does not improve the results, a bigger network is probably more convinient.

In TensorFlow, we have the `tf.nn.dropout()` function, which can be attached to a layer as follows:

```
keep_prob = tf.placeholder(tf.float32) # probability to keep units

hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
```

NOTE: We should only use dropout when training. When validating the model, we must use `keep_prob=1.0` to keep all the information.


