require "bundler/setup"
require 'tensor_stream'
require 'pry-byebug'

tf = TensorStream

rows = File.readlines(File.join("samples","iris.data")).map {|l| l.chomp.split(',') }

iris = rows[0...100]
iris.collect do |a, b, c, d, species|
  [a, b, c, d, species == 'Iris-setosa' ? 0 : 1]
end

srand(5)
seed = 5
tf.set_random_seed(seed)