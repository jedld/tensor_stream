require "bundler/setup"
require 'tensor_stream'
require 'pry-byebug'

tf = TensorStream

rows = File.readlines(File.join("samples","iris.data")).map {|l| l.chomp.split(',') }

iris = rows[0...100]
transformed_data = iris.collect do |a, b, c, d, species|
  [a, b, species == 'Iris-setosa' ? 0 : 1]
end

srand(5)
seed = 5
tf.set_random_seed(seed)

train_x = transformed_data[0..50].map { |x| x[0].to_f }
train_y = transformed_data[0..50].map { |x| x[1].to_f }

test_x = transformed_data[51..100].map { |x| x[0].to_f }
test_y = transformed_data[51..100].map { |x| x[1].to_f }

def map_norm(data)
  min = data.min
  max = data.max

  normalize = -> (val, high, low) {  (val - low) / (high - low) } # maps input to float between 0 and 1

  data.map do |x|
    normalize.(x, max, min)
  end
end

normalized_x = map_norm(train_x)
normalized_y = map_norm(train_y)

normalized_test_x = map_norm(test_x)
normalized_test_y = map_norm(test_y)

A = tf.variable(tf.random_normal(shape: [4, 1]))
b = tf.variable(tf.random_normal(shape: [1, 1]))

init = tf.global_variables_initializer
sess = tf.session
sess.run(init)

data = tf.placeholder(dtype: :float32, shape: [nil, 4])
target = tf.placeholder(dtype: :float32, shape: [nil, 1])

mod = tf.matmul(data, A) + b

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits: mod, labels: target))

learning_rate = 0.003
batch_size = 30
iter_num = 1500

optimizer = TensorStream::Train::GradientDescentOptimizer.new(learning_rate).minimize(loss)

prediction = tf.round(tf.sigmoid(mod))
# Bool into float32 type
correct = tf.cast(tf.equal(prediction, target), dtype: :float32)
# Average
accuracy = tf.reduce_mean(correct)