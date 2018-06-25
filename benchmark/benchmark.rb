require "bundler/setup"
require 'tensor_stream'
require 'benchmark'
require 'pry-byebug'
require 'awesome_print'
require 'tensor_stream/evaluator/opencl/opencl_evaluator'

def tr(t, places = 1)
  if t.is_a?(Array)
    return t.collect do |v|
      tr(v)
    end
  end

  return t unless t.is_a?(Float)

  t.round(places)
end

tf = TensorStream

srand(5)
seed = 5
tf.set_random_seed(seed)

SHAPES = [32, 32]

sess = tf.session(:ruby_evaluator)

a = tf.constant(sess.run(tf.random_uniform(SHAPES)))
a_int = tf.constant([
  [1, 2, 3, 4, 4, 1, 4, 8, 3, 4, 1, 1],
  [2, 2, 3, 4, 4, 1, 1, 1, 1, 4, 1, 1],
  [3, 2, 3, 4, 0, 1, 1, 2, 1, 1, 2, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 1, 1, 3, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 1, 1, 4, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 1, 5, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 1, 6, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 0, 0, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 2, 6, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 2, 1, 1],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 2, 1, 2],
  [4, 2, 3, 4, 0, 1, 1, 0, 0, 2, 1, 2],
])

b = tf.constant(sess.run(tf.random_uniform(SHAPES)))

c = tf.constant(sess.run(tf.random_uniform(SHAPES)))

d = tf.constant(sess.run(tf.random_uniform(SHAPES)))

p = tf.placeholder('float')
q = tf.placeholder('float')

model = -tf.sin(a.dot(b + p) + c).dot(a) + tf.cos(a.dot(d + q))
single_function_test = (tf.sigmoid(a * p) * tf.sigmoid(b * q)) + c
pow_f = tf.pow(a, 3)
pow_i = tf.pow(a_int, 3)
matmul = tf.matmul(a, b)
out_of_order = tf.matmul(a, b) + tf.matmul(a, c)
softmax = tf.nn.softmax(a)

puts TensorStream::Evaluator.default_evaluators

sess2 = tf.session

puts `cat /proc/cpuinfo | grep "model name" | head -1`
device = TensorStream::Evaluator::OpenclEvaluator.default_device.native_device
puts "OpenCL device #{device.platform.to_s} #{device.name}"
Benchmark.bmbm do |x|
  x.report("pure ruby ooo matmul     :") { 100.times do sess.run(out_of_order) end }
  x.report("opencl    ooo matmul     :") { 100.times do sess2.run(out_of_order) end }
  x.report("pure ruby softmax        :") { 100.times do sess.run(softmax) end }
  x.report("opencl    softmax        :") { 100.times do sess2.run(softmax) end }
  x.report("pure ruby matmul         :") { 100.times do sess.run(matmul) end }
  x.report("opencl    matmul         :") { 100.times do sess2.run(matmul) end }
  x.report("pure ruby                :") { 100.times do sess.run(model, feed_dict: { p => rand, q => rand }) end }
  x.report("opencl                   :") { 100.times do sess2.run(model, feed_dict: { p => rand, q => rand }) end }
  x.report("pure ruby single function:") { 100.times do sess.run(single_function_test, feed_dict: { p => rand, q => rand }) end }
  x.report("opencl     singlefunction:") { 100.times do sess2.run(single_function_test, feed_dict: { p => rand, q => rand }) end }
  x.report("pure ruby pow float:") { 100.times do sess.run(pow_f, feed_dict: { p => rand, q => rand }) end }
  x.report("opencl pow float:") { 100.times do sess2.run(pow_f, feed_dict: { p => rand, q => rand }) end }
  x.report("pure ruby pow int:") { 100.times do sess.run(pow_i, feed_dict: { p => rand, q => rand }) end }
  x.report("opencl pow int:") { 100.times do sess2.run(pow_i, feed_dict: { p => rand, q => rand }) end }
end