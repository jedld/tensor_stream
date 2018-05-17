require "bundler/setup"
require 'tensor_stream'
require 'pry-byebug'

tf = TensorStream

rows = File.readlines(File.join("samples","iris.data")).map {|l| l.chomp.split(',') }

iris = rows[0...100].shuffle!

transformed_data = iris.collect do |row|
  row[0, 4].map(&:to_f)
end

columns = (0..3).map do |i|
  transformed_data.map { |row| row[i] }
end

# Normalize data values before feeding into network
normalize = -> (val, high, low) { (val - low) / (high - low) } # maps input to float between 0 and 1

transformed_data.map! do |row|
  row.map.with_index do |val, j|
    max, min = columns[j].max, columns[j].min
    normalize.(val, max, min)
  end
end


srand(5)
seed = 5
tf.set_random_seed(seed)

train_x = transformed_data[0..50].map  { |x| x[0..3].map(&:to_f) }
train_y = iris[0..50].map  { |x| x[4] == 'Iris-setosa' ? 0.0 : 1.0 }

test_x = transformed_data[51..100].map { |x| x[0..3].map(&:to_f) }
test_y = iris[51..100].map { |x| x[4] == 'Iris-setosa' ? 0.0 : 1.0 }


A = tf.variable(tf.random_normal([4, 1]))
b = tf.variable(tf.random_normal([1, 1]))

init = tf.global_variables_initializer
sess = tf.session
sess.run(init)

data = tf.placeholder(:float32, shape: [nil, 4])
target = tf.placeholder(:float32, shape: [nil, 1])

mod = data.dot(A) + b

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits: mod, labels: target))

learning_rate = 0.003
batch_size = 30
iter_num = 1500

optimizer = TensorStream::Train::GradientDescentOptimizer.new(learning_rate)
goal = optimizer.minimize(loss)
prediction = tf.round(tf.sigmoid(mod))
# Bool into float32 type
correct = tf.cast(tf.equal(prediction, target), dtype: :float32)
# Average
accuracy = tf.reduce_mean(correct)

loss_trace = []
train_acc = []
test_acc = []

(0..iter_num).each do |epoch|
  batch_train_X = train_x
  batch_train_y = [train_y].transpose
  sess.run(goal, feed_dict: { data => batch_train_X, target => batch_train_y })
  temp_loss = sess.run(loss, feed_dict: {data => batch_train_X, target => batch_train_y})
end