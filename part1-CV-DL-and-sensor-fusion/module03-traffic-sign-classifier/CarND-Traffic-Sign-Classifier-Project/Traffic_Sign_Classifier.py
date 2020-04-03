#!/usr/bin/env python
# coding: utf-8

# # Traffic Sign Recognition Classifier
# ---
# ### Project from the third module of the Self-Driving Car Engineer Udacity's Nanodegree
# 
# [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
# 
# The aim of this project is to be able to classify traffic signs that appear in 32x32x3 RGB images that can be from 43 different classes using Deep Learning. In order to do so, I will be using the data from the [**German Traffic Sign Benchmark**](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) and will be basing my approach in the **LeNet-5** architecture. The quantified target is to go over a **93% of validation accuracy**. 
# 
# 
# # Step 0: Load The Data
# ---

# In[2]:


# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = '../data/train.p'
validation_file= '../data/valid.p'
testing_file = '../data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# # Step 1: Dataset summary, exploration and preparation
# ---
# ## Basic data summary

# In[3]:


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# DONE: Number of training examples
n_train = len(X_train)

# DONE: Number of validation examples
n_validation = len(X_valid)

# DONE: Number of testing examples.
n_test = len(X_test)

# DONE: What's the shape of an traffic sign image?
image_shape = X_train[0].shape # np.ndarray of 32x32x3

# DONE: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train)) # Numeric labels going from 0 to 42 (compute counting unique elements in training outputsDONE)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ## Data visualization
# ### Exploratory random data visualization
# Visualize one random traffic sign and its label. The correct association can be checked using the *signnames.csv* file of this repository, which relates the numerical labels with text descriptions.

# In[6]:


### Data visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random

# Get random index to plot from the training input set
index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

# Plot 10 input images and their labels
fig, axs = plt.subplots(2,5, figsize=(15, 6))
axs = axs.ravel()

for i in range(10):
    index = random.randint(0, len(X_train)) # Get random idx
    axs[i].axis('off') # Plot without axis
    axs[i].imshow(X_train[index]) #Show image
    axs[i].set_title(y_train[index]) #Show label as title


# ### Ammount of samples of each class
# It can be appreciated the big different between the ammount of samples of each class, which may difficult the classification of the classes with less examples (not now because I augmented the data to have more samples of the classes with lower representation).

# In[7]:


import numpy as np

unique_train, counts_train = np.unique(y_train, return_counts=True)
plt.bar(unique_train, counts_train)
plt.grid()
plt.title("Ammount of each class' samples - train data")
plt.show()


# ## Data pre-processing
# ### Pre-processing

# In[8]:


# Convert to grayscale
X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)
X_valid_gray = np.sum(X_valid/3, axis=3, keepdims=True)

print('Original dimensions: ' + str(X_train.shape))
print('Dimensions in grayscale: ' + str(X_train_gray.shape))
print('')

# Normalize data (forcing a zero mean) between -1 and 1, given that the pixel range is 0-255
X_train_norm = (X_train_gray-128)/128
X_valid_norm = (X_valid_gray-128)/128
X_test_norm = (X_test_gray-128)/128

print('Mean in grayscale: ' + str(np.mean(X_train)))
print('Mean after normalization: ' + str(np.mean(X_train_norm)))


# ## Data augmentation
# Given the huge lack of homogeneiety in the amount os samples of each class (some have more than 2000 examples and others just over 200), I will perform a data augmentation based on the one on [this](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) repository that will generate samples for the clasess represented by less than 750 examples until they have at least this ammount of samples.
# ### Random augmentation functions

# In[9]:


import cv2

def random_translate(img):
    rows,cols,_ = img.shape
    
    # allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)

    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst


def random_scaling(img):   
    rows,cols,_ = img.shape

    # transform limits
    px = np.random.randint(-2,2)

    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])

    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(rows,cols))
    
    dst = dst[:,:,np.newaxis]
    
    return dst


def random_warp(img):
    rows,cols,_ = img.shape

    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06   # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06

    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4

    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst


def random_brightness(img):
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    
    return dst


# ### Perform the data augmentation
# NOTE: This takes over an hour.

# In[7]:


print('Shapes before augmentation:')
print('  * Inputs: ' + str(X_train_norm.shape))
print('  * Labels: ' + str(y_train.shape))

input_indices = []
output_indices = []

# For each class
for class_n in range(n_classes):
    print(class_n, ': ', end='')
    class_indices = np.where(y_train == class_n)
    n_samples = len(class_indices[0])
    
    # If there are less than 750 samples of the current class
    if n_samples < 750:
        # Augmentate until we reach 750
        for i in range(750 - n_samples):
            input_indices.append(class_indices[0][i%n_samples])
            output_indices.append(X_train_norm.shape[0])
            new_img = X_train_norm[class_indices[0][i % n_samples]]
            new_img = random_translate(random_scaling(random_warp(random_brightness(new_img))))
            X_train_norm = np.concatenate((X_train_norm, [new_img]), axis=0)
            y_train = np.concatenate((y_train, [class_n]), axis=0)
            
            # Draw a vertical line each 50 generated samples
            if i % 50 == 0:
                print('|', end='')
            # Draw a dash each 10 generated samples
            elif i % 10 == 0:
                print('-',end='')
    print('')
        
print('Shapes after augmentation:')
print('  * Inputs: ' + str(X_train_norm.shape))
print('  * Labels: ' + str(y_train.shape))


# ### Plot histogram of the new examples and save the data

# In[10]:


# Plot histogram
unique_train, counts_train = np.unique(y_train, return_counts=True)
plt.bar(unique_train, counts_train)
plt.grid()
plt.title("Ammount of each class' samples - train data")
plt.show()

# Save augmented train data in a pickle file
pickle_dict = {'features':X_train_norm, 'labels':y_train}
pickle_out = open("augmented_train.p", "wb")
pickle.dump(pickle_dict, pickle_out)
pickle_out.close()


# ## Shuffle (the now augmented) data and split into train and validation
# To avoid influence the network with the order of the examples during training and validation.
# 
# I also want the validation set to be more homogeneus regarding the ammount of class examples.

# In[10]:


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Load augmented data
augmented_train_file = 'augmented_train.p'

with open(augmented_train_file, mode='rb') as f:
    ag_train = pickle.load(f)
    
X_train_norm, y_train = ag_train['features'], ag_train['labels']

# Shuffle data
X_train_norm, y_train = shuffle(X_train_norm, y_train)

# Split augmentation between train and validation
X_train, X_valid, y_train, y_valid = train_test_split(X_train_norm, y_train, test_size=0.20, random_state=42)

# Print the dimensions of the train
print('Size of the augmentated training set:', len(X_train_norm))
print('Size of the new training set:', len(X_train))
print('Size of the new validation set:', len(X_valid))


# # Step 2: Design and Test the Architecture
# ---
# ### Model Architecture
# #### Auxiliar function

# In[11]:


import tensorflow as tf

def get_conv_layer_elements(input_size, output_size, stride, padding, mu, sigma):
    """
    Returns the necessary elements to create a convolutional layer with the received restrictions.
    
    ::param input_size:: List with two elements: [in_height (or in_width), n_channels].
    ::param output_size::  List with two elements: [out_height (or out_width), n_outputs].
    ::param stride:: Integer with the stride that will be applied in both axes (outside this func).
    ::param padding:: String with the padding type ('SAME' or 'VALID').
    ::param mu:: Mean of the normal distribution to be sampled to initialise the weights
    ::param sigma:: Standard deviation of the normal distribution to be sampled to initialise the weights
    """
    
    int_padding = int(padding=='SAME')
    
    # Extract information from requirements
    filter_size = input_size[0] + 2*int_padding - stride*(output_size[0] -1) #Same width and height
    color_channels = input_size[1]
    k_output = output_size[1]
    
    multiD_stride = [1, stride, stride, 1]
    
    # Create TensorFlow weights and biases
    weights = tf.Variable(tf.truncated_normal([filter_size, filter_size, color_channels, k_output], mean=mu, stddev=sigma))
    biases = tf.Variable(tf.zeros(k_output))
    
    return weights, biases, multiD_stride


# ### Modified LeNet-5 architecture
# I use the modified version of the LeNet-5 network proposed in the [Sermanet/LeCunn's approach](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). They add an extra convolutional layer in parallel to the flattening of the last pooling that will be flattened and concatenated with the output of the other flattening. This gives an array of 800 elements that will be compressed toa 43-element representation by only one fully connected layer. These 43 elements will be the logits of the network.
# 
# I also use dropout to foment redundant learning to increase the network's robustness.

# In[12]:


from tensorflow.contrib.layers import flatten

def modified_LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    general_padding = 'VALID'
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    w1, b1, s1 = get_conv_layer_elements([32,1], [28,6], 1, general_padding, mu, sigma)
    conv_layer1 = tf.nn.conv2d(x, w1, s1, general_padding)
    conv_layer1 = tf.nn.bias_add(conv_layer1, b1)
    conv_layer1 = tf.nn.relu(conv_layer1)
    print("Layer 1 (conv)'s shape:", conv_layer1.get_shape())

    # Layer 2: Max Pooling. Input = 28x28x6. Output = 14x14x6.
    scale_fact = 2
    ksize = [1, scale_fact, scale_fact, 1]
    pool_layer2 = tf.nn.max_pool(conv_layer1, ksize, ksize, general_padding)
    print("Layer 2 (Mpool)'s shape:", pool_layer2.get_shape())
    
    # Layer 3: Convolutional. Output = 10x10x16.
    w3, b3, s3 = get_conv_layer_elements([14,6], [10,16], 1, general_padding, mu, sigma)
    conv_layer3 = tf.nn.conv2d(pool_layer2, w3, s3, general_padding)
    conv_layer3 = tf.nn.bias_add(conv_layer3, b3)
    conv_layer3 = tf.nn.relu(conv_layer3)
    print("Layer 3 (conv)'s shape:", conv_layer3.get_shape())

    # Layer 4: Max Pooling. Input = 10x10x16. Output = 5x5x16.
    scale_fact = 2
    ksize = [1, scale_fact, scale_fact, 1]
    pool_layer4 = tf.nn.max_pool(conv_layer3, ksize, ksize, general_padding)
    print("Layer 4 (Mpool)'s shape:", pool_layer4.get_shape())
    
    # Layer 5.1 (added path): Convolutional. Output = 1x1x400.
    w5, b5, s5 = get_conv_layer_elements([5,16], [1,400], 1, general_padding, mu, sigma)
    conv_layer5 = tf.nn.conv2d(pool_layer4, w5, s5, general_padding)
    conv_layer5 = tf.nn.bias_add(conv_layer5, b5)
    conv_layer5 = tf.nn.relu(conv_layer5)
    print("Layer 5 (conv)'s shape:", conv_layer5.get_shape())
    
    
    # Flatten output of layer 4. Input = 5x5x16. Output = 400.
    flat_pool_layer4 = flatten(pool_layer4)
    print("Flat_4's shape:", flat_pool_layer4.get_shape())
    
    # Flatten output of layer 5.1. Input = 1x1x400. Output = 400.
    flat_conv_layer5 = flatten(conv_layer5)
    print("Flat_5's shape:", flat_conv_layer5.get_shape())
    
    # Concatenate the flattened l4_pool and l51_conv. Input = 400 + 400. Output = 800
    concat_4_and_5 = tf.concat([flat_conv_layer5, flat_pool_layer4], 1)
    print("Concat_4_5's shape:", concat_4_and_5.get_shape())

    # Dropout (to avoid overfitting and foment redundant learning)
    dropout = tf.nn.dropout(concat_4_and_5, keep_prob)
    
    
    # Layer 6: Fully Connected. Input = 800. Output = 43.
    W6 = tf.Variable(tf.truncated_normal((800,43), mean = mu, stddev = sigma), name="W6")
    b6 = tf.Variable(tf.zeros(43), name="b6")    
    logits = tf.add(tf.matmul(dropout, W6), b6)
    print("Logits' shape:", logits.get_shape())
    
    return logits


# ## Train, Validate and Test the Model
# ### Setup

# In[13]:


# Hyperparameters
EPOCHS = 75
BATCH_SIZE = 100
LEARNING_RATE = 0.0009

# Input and output placeholders
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32) # Probability to keep weights in the dropout
one_hot_y = tf.one_hot(y, 43)


# ### Training pipeline

# In[14]:


logits = modified_LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)


# ### Evaluation pipeline and execution function

# In[15]:


# Evaluation pipeline
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

# Function to execut the evaluation given certain input and output data
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    
    # For each batch of data
    for offset in range(0, num_examples, BATCH_SIZE):
        # Get batch
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        
        # Run (keeping all the weights in the dropout))
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        
        # Sum accuracy to return the mean
        total_accuracy += (accuracy * len(batch_x))
        
    return total_accuracy / num_examples


# #### Training process

# In[14]:


with tf.Session() as sess:
    # List to store the validation accuracies for later plotting
    valid_accuracies = []
    
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    
    # For each EPOCH
    for i in range(EPOCHS):
        # Mix data (again)
        X_train, y_train = shuffle(X_train, y_train)
        
        # For each batch of data
        for offset in range(0, num_examples, BATCH_SIZE):
            # Get batch
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            
            # Run session (with a 75% of chance of keeping the weights in the dropout)
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
        
        # Run validation
        validation_accuracy = evaluate(X_valid, y_valid)
        valid_accuracies.append(validation_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    
    # Save ses
    saver.save(sess, './lenet')
    print("Model saved")
    
# Plot validation accuracy evolution
plt.plot(valid_accuracies)
plt.title("Evolution of the validation accuracy")
plt.show()


# ### Test the model with the testing set

# In[38]:


with tf.Session() as sess:
    # Load model
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    
    # Run test and output mean accuracy
    test_accuracy = evaluate(X_test_norm, y_test)
    print("Test accuracy = {:.3f}".format(test_accuracy))


# # Step 3: Test a Model on New Images
# ---
# I have chosen 5 new challenging images from the internet to analyze better the performance of the model. One is from a signal with an unusual not-outdoors environment, anotherone is a fake one with a dragon on it, the 3rd is a one normal one but slightly modified, the next one is a digital drawing and the last is an old scratched one.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.
# 
# ## Load and output the new images

# In[39]:


import glob
import matplotlib.pyplot as plt
import cv2

# Get list of new (resized) images
new_images = glob.glob('new_images/*')
print('Detected ' + str(len(new_images)) + ' new images: ' + str(new_images))

# Prepare plot
fig, axs = plt.subplots(1,len(new_images), figsize=(25, 25))
axs = axs.ravel()

# Read images, add them to lists and show them
new_images_orig = []
new_images_resized = []
for idx, im_name in enumerate(new_images):
    img = cv2.imread(im_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (32,32))
    
    new_images_orig.append(img)
    new_images_resized.append(img_resized)
    
    axs[idx].axis('off')
    axs[idx].imshow(img)


# ## Predict the Sign Type for Each Image
# ### Pre-process the images

# In[40]:


import numpy as np
# Transform from lists to np arrays
test_input = np.array(new_images_resized)

# Convert to grayscale
test_input = np.sum(test_input/3, axis=3, keepdims=True)

# Normalize data (forcing a zero mean) between -1 and 1, given that the pixel range is 0-255
test_input = (test_input-128)/128


# ### Load network and forward new samples

# In[42]:


import tensorflow as tf

test_labels = [1, 11, 25, 14, 18]
outputs = None # To store the outputted logits after forwarding the data

with tf.Session() as sess:
    # Load network
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    
    # Forward data to get logits
    outputs = sess.run(logits, feed_dict={x:test_input, keep_prob: 1.0})
    
    # Compute and print accuracy
    print('Accuracy: ' + str(evaluate(test_input, test_labels)))
    print('')

# Print the test labels
print('Test labels:')
print(test_labels)
print('')
    
# Create list of indexes of the top 3 resutlts
top_idxs = []
for i, output in enumerate(outputs):
    # Get higher scores
    idxs = output.argsort()[-3:][::-1]
    
    # Print TOP
    print('Top outputs of image ' + new_images[i])
    for j, score_idx in enumerate(idxs):
        print('  * TOP ' + str(j+1) + ': ' + str(score_idx))

