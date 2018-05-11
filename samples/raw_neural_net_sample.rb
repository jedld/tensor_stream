require "bundler/setup"
require 'tensor_stream'
require 'pry-byebug'

learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

tf = TensorStream

# tf Graph input
X = tf.placeholder("float", shape: [nil, num_input])
Y = tf.placeholder("float", shape: [nil, num_classes])

# Store layers weight & bias
@weights = {
    h1: tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    h2: tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    out: tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

@biases = {
    b1: tf.Variable(tf.random_normal([n_hidden_1])),
    b2: tf.Variable(tf.random_normal([n_hidden_2])),
    out: tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x)
    # Hidden fully connected layer with 256 neurons
    layer_1 =  TensorStream.add(TensorStream.matmul(x, @weights[:h1]), @biases[:b1])
    # Hidden fully connected layer with 256 neurons
    layer_2 = TensorStream.add(TensorStream.matmul(layer_1, @weights[:h2]), @biases[:b2])
    # Output fully connected layer with a neuron for each class
    TensorStream.matmul(layer_2, @weights[:out]) + @biases[:out]
end

def softmax(logits)
  TensorStream.exp(logits) / TensorStream.reduce_sum(TensorStream.exp(logits))
end

# Construct model
logits = neural_net(X)
prediction = softmax(logits)

puts prediction.to_math