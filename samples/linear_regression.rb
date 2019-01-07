require 'tensor_stream'

tf = TensorStream

learning_rate = 0.01
training_epochs = 1000
display_step = 50

train_x = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
            7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]

train_y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
            2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]

n_samples = train_x.size

x_value = Float.placeholder
y_value = Float.placeholder

# Set model weights
weight = rand.t.var name: "weight"

bias = rand.t.var name: "bias"

# Construct a linear model
pred = x_value * weight + bias

# Mean squared error
cost = ((pred - y_value)**2).reduce / (2 * n_samples)

# Other optimizers --
#
# optimizer = TensorStream::Train::MomentumOptimizer.new(learning_rate, momentum, use_nesterov: true).minimize(cost)
# optimizer = TensorStream::Train::AdamOptimizer.new(learning_rate).minimize(cost)
# optimizer = TensorStream::Train::AdadeltaOptimizer.new(1.0).minimize(cost)
# optimizer = TensorStream::Train::AdagradOptimizer.new(0.01).minimize(cost)
# optimizer = TensorStream::Train::RMSPropOptimizer.new(0.01, centered: true).minimize(cost)
optimizer = TensorStream::Train::GradientDescentOptimizer.new(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer

tf.session do |sess|
  start_time = Time.now
  sess.run(init)

  (0..training_epochs).each do |epoch|
    train_x.zip(train_y).each do |x, y|
      sess.run(optimizer, feed_dict: { x_value => x, y_value => y })
    end

    if (epoch + 1) % display_step == 0
      c = sess.run(cost, feed_dict: { x_value => train_x, y_value => train_y })
      puts("Epoch:", '%04d' % (epoch + 1), "cost=", c, \
           "W=", sess.run(weight), "b=", sess.run(bias))
    end
  end

  puts "Optimization Finished!"
  training_cost = sess.run(cost, feed_dict: { x_value => train_x, y_value => train_y })
  puts "Training cost=", training_cost, "W=", sess.run(weight), "b=", sess.run(bias), '\n'
  puts "time elapsed ", Time.now.to_i - start_time.to_i
end