#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Solution is available in the other "solution.ipynb" 
import tensorflow as tf


def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    
    # DONE: Calculate the softmax of the logits
    softmax = tf.nn.softmax(logits)
    
    with tf.Session() as sess:
        pass
        # DONE: Feed in the logit data
        output = sess.run(softmax, feed_dict={logits:logit_data})

    return output


# In[2]:


### DON'T MODIFY ANYTHING BELOW ###
### Be sure to run all cells above before running this cell ###
import grader

try:
    grader.run_grader(run)
except Exception as err:
    print(str(err))


# In[ ]:




