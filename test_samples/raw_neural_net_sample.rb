""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
require "bundler/setup"
require 'tensor_stream'
require 'mnist-learn'
# require 'tensor_stream/evaluator/opencl/opencl_evaluator'
require 'pry-byebug'

tf = TensorStream
# Import MNIST data
mnist = Mnist.read_data_sets('/tmp/data', one_hot: true)

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder(:float64, shape: [nil, num_input])
Y = tf.placeholder(:float64, shape: [nil, num_classes])

# Store layers weight & bias
weights = {
    'h1' => tf.variable(tf.random_normal([num_input, n_hidden_1]), dtype: :float64, name: 'h1'),
    'h2' => tf.variable(tf.random_normal([n_hidden_1, n_hidden_2]), dtype: :float64, name: 'h2'),
    'out' => tf.variable(tf.random_normal([n_hidden_2, num_classes]), dtype: :float64, name: 'out')
}

biases = {
    'b1' => tf.variable(tf.random_normal([n_hidden_1]), dtype: :float64, name: 'b1'),
    'b2' => tf.variable(tf.random_normal([n_hidden_2]), dtype: :float64, name: 'b2'),
    'out' => tf.variable(tf.random_normal([num_classes]), dtype: :float64, name: 'out2')
}


# Create model
def neural_net(x, weights, biases)
    tf = TensorStream
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    tf.matmul(layer_2, weights['out']) + biases['out']
end

# Construct model
logits = neural_net(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits: logits, labels: Y))

optimizer = TensorStream::Train::GradientDescentOptimizer.new(learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, :float32))

# tf.add_check_numerics_ops

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer

# Start training
tf.session do |sess|
    # Run the initializer
    sess.run(init)

    (1..num_steps+1).each do |step|
        
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict: { X => batch_x, Y => batch_y })
        if step % display_step == 0 || step == 1
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict: { X => batch_x, Y => batch_y})
        print("Step " + step.to_s + ", Minibatch Loss= " + \
                loss.to_s + ", Training Accuracy= " + \
                acc.to_s)
        end
    end

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict: {  X => mnist.test.images,
                                        Y => mnist.test.labels}))
end