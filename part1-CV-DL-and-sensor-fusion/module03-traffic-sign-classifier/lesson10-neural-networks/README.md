# Lesson 10. Neural Networks

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Linear to logistic regression

The two main problems of Machine learning are:
- **Regression**: Predict values on a continuous space. i.e. given several pairs (or sets) of associated data (for this X value, the output is this Y one), we can perform linear regression to predict the Y value that corresponds to a new unseen X input.
- **Classification**: When working with discrete classes (e.g. knowing who is talking on a conference talk, identifying fish species, etc.). For this, we can use logistic regression.


We can fit models of the form *y = Wx+b*, where *W* is a weight vector (w1, w2, ..., wn) and *x* is the input vector (x1, x2, ..., xn). The *b* is the bias and the *y* the output. *ŷ* will be the predicted class. We will have a positive result (*ŷ=1*) if the output is positive (*y>=0*) and a negative one (*ŷ=0*) if the output is negative (*y<0*).

We can add more data adding more dimensions to the input space. The boundary separating the output classes for a n-dimensional input space will be a n-1 hyperplane.


## Perceptron

A perceptron is a node in a computation graph that evaluates a certain condition. It has N inputs for data that will be weighted by N independent weigths and modified by a bias. It will output the value of an *activation function* in the point outputted by the *Wx+b* function. For example, if the chosen activation function is a ReLu, the perceptron output will be *y = ReLu(Wx+b)*. We can model the bias as the weight of an extra input node which input is always *1*.

There are several options for the **activation function**, the simplest one is the *step function*. If X is below 0, it will output 0 and, if it is over 0, the output will be 1.

The perceptrons can be connected in *layers* of a neural network, in a way that the outputs of one layer of perceptrons will be forwarded as inputs of the perceptrons of the next layer.


## Weights adjustement

The values of the weights give more or less relevant each piece of input data for the final decision. These weights are adjusted during the *learning* process of the (perceptron) networks. Using examples of data (output labels connected to sets of inputs), they will *learn* which input should be more important fo the final decision.

The weights start being random and the *training* step modifies them using the *training data* (labels connected with inputs). In this phase, the **inputs are forwarded** trough the network, the **output is compared** with the true one in the training data, an **error** is computed and this error is ***backpropagated*** to correct the weights in a way that the network will behave better if it receives a similar input set.

The bias makes the perceptrons have a bigger tendence to provide positive or negative outputs.

The weight update is computed using the used error and the *learning rate* (a factor that will smooth the corrections in order to modify smoothly the mathmeatical model so that the fitting will not be drastic for each missclassified point each epoch). A way of updating a line that classifies a problem with bi-dimensional input is substracting from its parameters (w1, w2 and b) the coordinates of the missclassified input adding a 1. This will be:

- w1' = w1-x1
- w2' = w2-x2
- b' = b-1

That, using a sample learning rate hyperparameter of 0.1, will become:

- w1' = w1-x1*0.1
- w2' = w2-x2*0.1
- b' = b-0.1

Note: we will add or substract the point coordinates depending on if the point is in the negative or positive side of the boundary defined by the model.

We can stablish several termination criterias for this process, such as *perform X times*, *perform until there are less than X missclassified points*, *perform until there is no missclassified point*, etc.


## Error functions

In order to correct our predictions, we need to stablish an error function that will give us an idea of how wrong a missclasified point is. Using them, we will be able to not only model what do we want to maximize/minimize but which things are more relevant for our problem.

One posible error functions is counting the ammount of missclassified points. The problem with this is that we will have a discrete error space, so many times, wherever we look in our neighbourhood, we will see no variation from our current situation (given that the correction steps are usually pretty small).

This makes evident the advantage of a continuous error function. It is also convinient for it to be differentiable in order to be able to apply gradient descent. An example of a continuous error function is one that assigns a small penalty to the correctly classified points and a large one to the missclassified ones. Adding these penalties will make it possible to modify the boundary model in a way that makes the global penalty smaller and iterate until it is minimized.

In order to use continuous error functions we will need a **continuous output space**. That means to move from the *yes or no* approach to a probabilistic one (e.g. *68.2% likely*). We can make this change by moving from the step activation function to the **sigmoid activation function**.


## Multi-class classification

When we have more than two classes (color, animal specie, etc.), we will use the *softmax function*, which is a 3+ dimensional equivalent of the sigmoid. Once having the scores for each input, we need them to be probabilities that add a total of one, keeping the *ranking*. We can divide each score by the sum of all of them, but it will give us problems with negative scores, such as divisions by 0. We can turn all the scores into positive using the **exponential function** of the score and performing the same division. We will lose the proportionality between the scores but still keep the ranking. This is called the **softmax function**.


## One-hot encoding

There are sometimes in which we can not encode our inputs/outputs as numbers. In these cases, we could use only one variable and assign each of its values to each class of our problem. This, however, will lead to having more possible values than classes. This is why the chosen option is the **one-hot encoding**, which consist in creating one variable for each class, so that we will have several outputs in our system and each one of them will give us the probability of the input being each output class (e.g. 0.5 of being a dog, 0.3 of being a cat, 0.1 of being a duck and 0.1 of being a fish).


## Maximum likelihood and cross-entropy

When we have two models, we have to decide which one performs better, we will tell that the one closer to the actual outputs is better. i.e. if one model said that the output will be one with a 0.8 likelihood and another one sais 0.55 likelihood, if the output has to be one, the first is better, even both of them said it was more likely to be 1 than 0. If the output had to be 0, the second model is *less wrong* than the first one, so it would have been the best.

To compute this with several samples, we can assume independency between the probabilites, so we can multiply the probabilities of each point (input sample) being of its actual class according the model. The result of this multiplication will be the **maximum likelihood** and the higher it is, the better the model is. However, multiplying values between 0 and 1 will lead to very tiny results that may be problematic for a digital representation. In order to solve this, we can use logarithms to transform multiplications into sums (*log(ab) = log(a) + log(b)*).

By convention, the natural logarithm is used and, given that the logarithm of a number between 0 and 1 is always negative, we can substract them to get possitive scores. This sum of the negatives of the probabilities is called **cross-entropy**, which will be smaller for better models.

This will get high values from missclassified points and small values from the properly classified ones. This can therefore be used as an **error measure**.

When we deal with multi-class problems, we will need to use a generic fourmula.


## Gradient descent

We can compute the (negative) gradient wherever we are on the error function so that we will get the direction towards which we should move our function in order to minimize the error. This gradient will be the vector formed by the paratial derivatives of the error function respect to the each weight (and the bias). We will repeat this process in order to minimize the error. We will multiply the gradient by the learning rate (*alpha*) in order to avoid dramatic changes. The new weight will be the previous one minus alpha times the partial derivative of the error with respect to the weight that is being updated.

The farther away the prediction is from the actual label, the bigger will be the gradient, forcing the correction to be bigger as well.


## Non-linear models

A lot of the data in the real world can not be classified using a linear model. To create non-linear models we need to use several perceptrons or, in other words, create a perceptron network, also known as artificial neural network.

There are two ways in which we can add complexity to our models. (1) Adding more neurons to our layers and (2) adding more layers. Adding more neurons on the input layer means more dimensions in our linear model. With more neurons in our output layer, we will have more outputs, meaning that we will be able to classify between more classes. Adding more layers means combining linear models to create more and more non-linear models to fit more and more complex data distributions.

For multi-class neural networks we will need to add a softmax layer that distributes the scores in probabilities that will add 1.


## Backpropagation

This method is the one used to correct the weights of a neural network once the prediction and its error have been computed. It will modify the weights of the deepest layers to listen more or less to each previous node (linear or non-linear model) and the weights of the first layers to adjust the models better to the data.

