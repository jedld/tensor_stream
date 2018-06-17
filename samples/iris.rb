require "bundler/setup"
require 'tensor_stream'

# This neural network will predict the species of an iris based on sepal and petal size
# Dataset: http://en.wikipedia.org/wiki/Iris_flower_data_set
tf = TensorStream
rows = File.readlines(File.join("samples","iris.data")).map {|l| l.chomp.split(',') }

rows.shuffle!

label_encodings = {
  'Iris-setosa'     => [1, 0, 0],
  'Iris-versicolor' => [0, 1, 0],
  'Iris-virginica'  => [0, 0, 1]
}

x_data = rows.map {|row| row[0,4].map(&:to_f) }
y_data = rows.map {|row| label_encodings[row[4]] }

# Normalize data values before feeding into network
normalize = -> (val, high, low) {  (val - low) / (high - low) } # maps input to float between 0 and 1

columns = (0..3).map do |i|
  x_data.map {|row| row[i] }
end

x_data.map! do |row|
  row.map.with_index do |val, j|
    max, min = columns[j].max, columns[j].min
    normalize.(val, max, min)
  end
end

x_train = x_data.slice(0, 100)
y_train = y_data.slice(0, 100)

x_test = x_data.slice(100, 50)
y_test = y_data.slice(100, 50)

test_cases = []
x_train.each_with_index do |x, index|
  test_cases << [x, y_train[index]]
end

validation_cases = []
x_test.each_with_index do |x, index|
  validation_cases << [x, y_test[index] ]
end

learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 4 # 1st layer number of neurons
num_classes = 3 # MNIST total classes (0-9 digits)
num_input = 4
training_epochs = 100

tf = TensorStream

# tf Graph input
x = tf.placeholder("float", shape: [nil, num_input], name: 'x')
y = tf.placeholder("float", shape: [nil, num_classes], name: 'y')

# Store layers weight & bias
weights = {
  h1: tf.variable(tf.random_normal([num_input, n_hidden_1]), name: 'h1'),
  out: tf.variable(tf.random_normal([num_classes, num_classes]), name: 'out')
}

biases = {
  b1: tf.variable(tf.random_normal([n_hidden_1]), name: 'b1'),
  out: tf.variable(tf.random_normal([num_classes]), name: 'b_out')
}

# Create model
def neural_net(x, weights, biases)
    layer_1 = TensorStream.tanh(TensorStream.add(TensorStream.matmul(x, weights[:h1]), biases[:b1], name: 'layer1_add'))
    # Output fully connected layer with a neuron for each class
    TensorStream.sigmoid(TensorStream.matmul(layer_1, weights[:out]) + biases[:out])
end

# Construct model
logits = neural_net(x, weights, biases)

# Mean squared error
cost = TensorStream.reduce_sum(TensorStream.pow(logits - y, 2)) / ( 2 * y_train.size)
optimizer = TensorStream::Train::GradientDescentOptimizer.new(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = TensorStream.global_variables_initializer()

TensorStream.session do |sess|
  puts "init vars"
  sess.run(init)
  puts "Testing the untrained network..."
  loss = sess.run(cost, feed_dict: { x => x_train, y => y_train })
  puts loss
  puts "loss before training"
  (0..training_epochs).each do |epoch|
    x_train.zip(y_train).each do |t_x, t_y|
      sess.run(optimizer, feed_dict: { x => [t_x], y => [t_y] })
      loss = sess.run(cost, feed_dict: { x => [t_x], y => [t_y] })
    end
    puts "loss #{loss}"
  end
  loss = sess.run(cost, feed_dict: { x => x_train, y => y_train })
  puts "loss after training #{loss}"
end