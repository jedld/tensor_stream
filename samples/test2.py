import tensorflow as tf

x = tf.constant([[1.0, 0.5, 4.0]])

w = tf.constant([[0.4, 0.2],[0.1, 0.45],[0.2, 4.0]])

w2 = tf.constant([[0.3, 0.2],[0.15, 0.45]])
w3 = tf.constant([[0.1, 0.1, 1.0, 1.1, 0.4],[0.05, 0.2, 1.0, 1.2, 0.5],])

b= tf.constant([4.0, 5.0])
b2= tf.constant([4.1, 5.1])
b3 = tf.constant([2.0, 3.1, 1.0, 0.2, 0.2])

a = tf.sin(tf.matmul(x, w) + b)
a2 = tf.sin(tf.matmul(a, w2) + b2)
a3 = tf.tanh(tf.matmul(a2, w3) + b3)

g = tf.gradients(a3, [w, b])