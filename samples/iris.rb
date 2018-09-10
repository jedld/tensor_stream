require "bundler/setup"
require 'tensor_stream'
# require 'tensor_stream/evaluator/opencl/opencl_evaluator'

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



def init_weights(shape)
  # Weight initialization
  weights = TensorStream.random_normal(shape, stddev: 0.1)
  TensorStream.variable(weights)
end

def forwardprop(x, w_1, w_2)
  # Forward-propagation.
  # IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
  h  = TensorStream.nn.sigmoid(TensorStream.matmul(x, w_1))  # The \sigma function
  TensorStream.matmul(h, w_2)  # The \varphi function
end

x_size = x_train[0].size
y_size = y_train[0].size
h_size = 256
X = tf.placeholder(:float32, shape: [nil, x_size])
y = tf.placeholder(:float32, shape: [nil, y_size])

# Weight initializations
w_1 = init_weights([x_size, h_size])
w_2 = init_weights([h_size, y_size])

# Forward propagation
yhat    = forwardprop(X, w_1, w_2)
predict = tf.argmax(yhat, 1)

# Backward propagation
cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels: y, logits: yhat))

# updates =  TensorStream::Train::GradientDescentOptimizer.new(0.01).minimize(cost)
# updates =  TensorStream::Train::MomentumOptimizer.new(0.01, 0.5, use_nesterov: true).minimize(cost)
updates =  TensorStream::Train::RMSPropOptimizer.new(0.01).minimize(cost)

# Run SGD
sess = tf.session
init = tf.global_variables_initializer
sess.run(init)
loss = sess.run(cost, feed_dict: { X => x_test, y => y_test })
puts "loss test data set #{loss}"
loss = sess.run(cost, feed_dict: { X => x_train, y => y_train })
puts "Testing the untrained network..."
puts loss
start_time = Time.now
(0..100).each do |epoch|
  x_train.size.times do |i|
    sess.run(updates, feed_dict: {X => [x_train[i]], y => [y_train[i]]})
  end

  loss = sess.run(cost, feed_dict: { X => x_train, y => y_train })
  puts "epoch: #{epoch}, loss #{loss}"
end

loss = sess.run(cost, feed_dict: { X => x_train, y => y_train })
puts "loss after training #{loss}"
loss = sess.run(cost, feed_dict: { X => x_test, y => y_test })
puts "loss test data set #{loss}"
puts("time elapsed ", Time.now.to_i - start_time.to_i)