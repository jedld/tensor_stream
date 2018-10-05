# A ruby port of the example code discussed by Martin Gorner in
# "TensorFlow and Deep Learning without a PhD, Part 1 (Google Cloud Next '17)""
#
# https://www.youtube.com/watch?v=u4alGiomYP4
#
# Requirements:
#   mnist-learn gem
#   opencl_ruby_ffi gem
require "bundler/setup"
require 'tensor_stream'
require 'mnist-learn'

# Enable OpenCL hardware accelerated computation, not using OpenCL can be very slow
# require 'tensor_stream/opencl'

tf = TensorStream

# Import MNIST data
puts "downloading minst data"
mnist = Mnist.read_data_sets('/tmp/data', one_hot: true)
puts "downloading finished"

x = tf.placeholder(:float32, shape: [nil, 784])
w = tf.variable(tf.zeros([784, 10]))
b = tf.variable(tf.zeros([10]))



# model
y = tf.nn.softmax(tf.matmul(tf.reshape(x, [-1, 784]), w) + b)

y_ = tf.placeholder(:float32, shape: [nil, 10])

# loss function
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy =  tf.reduce_mean(tf.cast(is_correct, :float32))

optimizer = TensorStream::Train::AdamOptimizer.new
train_step = optimizer.minimize(cross_entropy)

sess = tf.session
init = tf.global_variables_initializer
sess.run(init)

(0...1000).each do |i|
  # load batch of images and correct answers
  batch_x, batch_y = mnist.train.next_batch(100)
  train_data = { x => batch_x, y_ => batch_y }

  # train
  sess.run(train_step, feed_dict: train_data)
  if (i % 10 == 0)
    # success? add code to print it
    a, c = sess.run([accuracy, cross_entropy], feed_dict: train_data)
    puts "#{i} train accuracy #{a}, error #{c}"

    # success on test data?
    test_data = { x => mnist.test.images, y_ => mnist.test.labels }
    a, c = sess.run([accuracy, cross_entropy], feed_dict: test_data)
    puts " test accuracy #{a}, error #{c}"
  end
end

