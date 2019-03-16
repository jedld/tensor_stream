# A ruby port of the example code discussed by Martin Gorner in
# "TensorFlow and Deep Learning without a PhD, Part 1 (Google Cloud Next '17)""
#
# https://www.youtube.com/watch?v=u4alGiomYP4
#
# Requirements:
#   mnist-learn gem
#   opencl_ruby_ffi gem
require "bundler/setup"
require "tensor_stream"
require "mnist-learn"

# Enable OpenCL hardware accelerated computation, not using OpenCL can be very slow
# gem install tensor_stream-opencl
require 'tensor_stream/opencl'

tf = TensorStream

# Import MNIST data
puts "downloading minst data"
mnist = Mnist.read_data_sets("/tmp/data", one_hot: true)
puts "downloading finished"