require "bundler/setup"
require "tensor_stream"
require "tensor_stream/evaluator/opencl/opencl_evaluator"
# require 'pry-byebug'

ts = TensorStream

n = 10
DIMEN = 1024

A = ts.random_uniform([DIMEN, DIMEN]).eval
B = ts.random_uniform([DIMEN, DIMEN]).eval

# Create a graph to store results
c1 = []
c2 = []
a = nil
b = nil

def matpow(m, n)
  return m if n < 1
  TensorStream.matmul(m, matpow(m, n - 1))
end

ts.device("/device:GPU:0") do
  a = ts.placeholder(:float32, shape: [DIMEN, DIMEN])
  b = ts.placeholder(:float32, shape: [DIMEN, DIMEN])
  # Compute A^n and B^n and store results in c1
  c1 << matpow(a, n)
  c1 << matpow(b, n)
end

sum = ts.device("/device:GPU:0") {
  ts.add_n(c1)
}

t1_1 = Time.now.to_i
t2_1 = nil

ts.session(log_device_placement: true) do |sess|
  sess.run(sum, feed_dict: {a => A, b => B})
  t2_1 = Time.now.to_i
end

# Multi GPU computing
# GPU:0 computes A^n
ts.device("/device:GPU:1") do
  a = ts.placeholder(:float32, shape: [DIMEN, DIMEN])
  c2 << matpow(a, n)
end

# GPU:1 computes B^n
ts.device("/device:GPU:1") do
  b = ts.placeholder(:float32, shape: [DIMEN, DIMEN])
  c2 << matpow(b, n)
end

ts.device("/device:GPU:1") do
  sum = ts.add_n(c2) # Addition of all elements in c2, i.e. A^n + B^n
end

t1_2 = Time.now.to_i
t2_2 = nil
ts.session(log_device_placement: true) do |sess|
  # Run the op.
  sess.run(sum, feed_dict: {a => A, b => B})
  t2_2 = Time.now.to_i
end

print("Single GPU computation time: " + (t2_1 - t1_1).to_s)
print("Multi GPU computation time: " + (t2_2 - t1_2).to_s)
