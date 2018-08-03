'''
A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
require "bundler/setup"
require 'tensor_stream'
require 'mnist-learn'
require 'tensor_stream/evaluator/opencl/opencl_evaluator'

tf = TensorStream

# Import MNIST data
mnist = Mnist.read_data_sets('/tmp/data', one_hot: true)

# In this example, we limit mnist data
Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200) #200 for testing

# tf Graph Input
xtr = tf.placeholder(:float, shape: [nil, 784])
xte = tf.placeholder(:float, shape: [784])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), 1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.argmin(distance, 0)

accuracy = 0.0

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
tf.session(:opencl_evaluator) do |sess|
  # Run the initializer
  sess.run(init)
  Xte.size.times do |i|
    # Get nearest neighbor
    nn_index = sess.run(pred, feed_dict: {xtr => Xtr, xte => Xte[i]})
    print("Test ", i, "Prediction: ",Ytr[nn_index].max, \
            "True Class: ", Yte[i].max, "\n")
    if Ytr[nn_index].max == Yte[i].max
      accuracy += 1.0/ Xte.size
    end
  end

  print("Done!")
  print("Accuracy:", accuracy)
end