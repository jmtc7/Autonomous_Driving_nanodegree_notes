# Solution is available in the "solution.ipynb" 
import tensorflow as tf

# Convert "(10/2)-1" to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
aux_const = tf.constant(1, tf.float64)
z = tf.divide(x, y)
z = tf.subtract(z, aux_const)

with tf.Session() as sess:
    output = sess.run(z)
    print(output)
