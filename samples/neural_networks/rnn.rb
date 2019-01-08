# RNN sample
#
# Ruby port Example based on article by Erik Hallström
# https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
#
#

require "bundler/setup"
require 'tensor_stream'
# require 'tensor_stream/opencl'
# require 'pry-byebug'


tf = TensorStream

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length / batch_size / truncated_backprop_length
randomizer = TensorStream.random_uniform([total_series_length], minval: 0, maxval: 2)


def generate_data(randomizer, total_series_length, batch_size, echo_step)
  x = randomizer.eval
  y = x.rotate(-echo_step)

  y[echo_step] = 0

  x = TensorStream::TensorShape.reshape(x, [batch_size, -1])  # The first index changing slowest, subseries as rows
  y = TensorStream::TensorShape.reshape(y, [batch_size, -1])
  [x, y]
end

batchX_placeholder = tf.placeholder(:float32, shape: [batch_size, truncated_backprop_length], name: 'batch_x')
batchY_placeholder = tf.placeholder(:int32, shape: [batch_size, truncated_backprop_length], name: 'batch_y')

init_state = tf.placeholder(:float32, shape: [batch_size, state_size], name: 'init_state')


W = tf.variable(tf.random_uniform([state_size+1, state_size]), dtype: :float32, name: 'W')
b = tf.variable(tf.zeros([state_size]), dtype: :float32, name: 'b')

W2 = tf.variable(tf.random_uniform([state_size, num_classes]), dtype: :float32, name: 'W2')
b2 = tf.variable(tf.zeros([num_classes]), dtype: :float32, name: 'b2')


inputs_series = tf.unpack(batchX_placeholder, axis: 1)
labels_series = tf.unpack(batchY_placeholder, axis: 1)

current_state = init_state
states_series = []

inputs_series.each do |current_input|
  current_input = tf.reshape(current_input, [batch_size, 1])
  input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # Increasing number of columns
  next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
  states_series << next_state
  current_state = next_state
end

logits_series = states_series.collect do |state|
  tf.matmul(state, W2) + b2
end

predictions_series = logits_series.collect do |logits|
  tf.nn.softmax(logits)
end

losses = logits_series.zip(labels_series).collect do |logits, labels|
  tf.nn.sparse_softmax_cross_entropy_with_logits(logits: logits, labels: labels)
end
total_loss = tf.reduce_mean(losses)

train_step = TensorStream::Train::AdagradOptimizer.new(0.1).minimize(total_loss)

puts "#{tf.get_default_graph.nodes.keys.size} nodes created"
zeros_state = tf.zeros([batch_size, state_size]).eval
tf.session do |sess|
  sess.run(tf.global_variables_initializer)
  (0..num_epochs).each do |epoch_idx|
    x, y = generate_data(randomizer, total_series_length, batch_size, echo_step)
    _current_state = zeros_state
    print("New data, epoch", epoch_idx, "\n")
    (0..num_batches - 1).each do |batch_idx|
      start_idx = batch_idx * truncated_backprop_length
      end_idx = start_idx + truncated_backprop_length

      batchX = x.map { |x| x[start_idx...end_idx] }
      batchY = y.map { |y| y[start_idx...end_idx] }

      _total_loss, _train_step, _current_state, _predictions_series = sess.run(
          [total_loss, train_step, current_state, predictions_series],
          feed_dict: {
              batchX_placeholder => batchX,
              batchY_placeholder => batchY,
              init_state => _current_state
          })

      if batch_idx%10 == 0
          print("Step",batch_idx, " Loss ", _total_loss, "\n")
      end
    end
  end
end