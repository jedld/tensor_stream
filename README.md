[![Gem Version](https://badge.fury.io/rb/tensor_stream.svg)](https://badge.fury.io/rb/tensor_stream)

[![CircleCI](https://circleci.com/gh/jedld/tensor_stream.svg?style=svg)](https://circleci.com/gh/jedld/tensor_stream)

# TensorStream

A reimplementation of TensorFlow for ruby. This is a ground up implementation with no dependency on TensorFlow. Effort has been made to make the programming style as near to TensorFlow as possible, comes with a pure ruby evaluator by default with support for an opencl evaluator for large models and datasets.

The goal of this gem is to have a high performance machine learning and compute solution for ruby with support for a wide range of hardware and software configuration.

## Features

- Replicates most of the commonly used low-level tensorflow ops (tf.add, tf.constant, tf.placeholder, tf.matmul, tf.sin etc...)
- Supports auto-differentiation via tf.gradients (mostly)
- Provision to use your own opcode evaluator (opencl, sciruby and tensorflow backends planned)
- Goal is to be as close to TensorFlow in behavior but with some freedom to add ruby specific enhancements (with lots of test cases)
- eager execution (experimental)

Since this is a pure ruby implementation for now, performance is not there yet. However it should be a good enough environment to learn about tensorflow and experiment with some models.

## Installation

Installation is easy, no need to mess with docker, python, clang or other shennanigans, works with both mri and jruby out of the box.

Add this line to your application's Gemfile:

```ruby
gem 'tensor_stream'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install tensor_stream

## Usage

Usage is similar to how you would use TensorFlow except with ruby syntax

Linear regression sample:

```ruby
require 'tensor_stream'

tf = TensorStream

learning_rate = 0.01
training_epochs = 1000
display_step = 50

train_X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
7.042,10.791,5.313,7.997,5.654,9.27,3.1]
train_Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
2.827,3.465,1.65,2.904,2.42,2.94,1.3]

n_samples = train_X.size

X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.variable(rand, name: "weight")
b = tf.variable(rand, name: "bias")

# Construct a linear model
pred = X * W + b

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / ( 2 * n_samples)

optimizer = TensorStream::Train::GradientDescentOptimizer.new(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

tf.session do |sess|
    start_time = Time.now
    sess.run(init)
    (0..training_epochs).each do |epoch|
      train_X.zip(train_Y).each do |x,y|
        sess.run(optimizer, feed_dict: {X => x, Y => y})
      end

      if (epoch+1) % display_step == 0
        c = sess.run(cost, feed_dict: { X => train_X, Y => train_Y })
        puts("Epoch:", '%04d' % (epoch+1), "cost=",  c, \
            "W=", sess.run(W), "b=", sess.run(b))
      end
    end

    puts("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict: { X => train_X, Y => train_Y})
    puts("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    puts("time elapsed ", Time.now.to_i - start_time.to_i)
end
```

You can take a look at spec/tensor_stream/operation_spec.rb for a list of supported ops and various examples and test cases used. Of course these contain only a
sliver of what TensorFlow can do, so feel free to file a PR to add requested
ops and test cases.

Other working samples can also be seen under tensor_stream/samples.

Samples that are used for development and are still being made to work can be found under test_samples

## Python to Ruby guide

Not all ops are available. Available ops are defined in lib/tensor_stream/ops.rb, corresponding gradients are found at lib/tensor_stream/math_gradients.

There are also certain differences with regards to naming conventions, and named parameters:

# Variables

To make referencing python examples easier it is recommended to use "tf" as the TensorStream namespace

At the beginning
```ruby
tf = TensorStream # recommended to use tf since most sample models on the net use this
ts = TensorStream # use this if you plan to use TensorStream only features, so other devs will know about that
```

Note the difference in named and optional parameters

Python

```python
w = ts.Variable(0, name='weights')
w = ts.Variable(0, 'weights')
```

Ruby

```ruby
w = ts.variable(0, name: 'weights')
```

# Shapes

Python
```python
x = tf.placeholder(tf.float32, shape=(1024, 1024))
x = tf.placeholder(tf.float32, shape=(None, 1024))
```

ruby supports symbols for specifying data types, nil can be used for None

Ruby
```ruby
x = ts.placeholder(:float32, shape: [1024, 1024])
x = ts.placeholder(:float32, shape: [nil, 1024])
```

For debugging, each operation or tensor supports the to_math method

```ruby
X = ts.placeholder("float")
Y = ts.placeholder("float")
W = ts.variable(rand, name: "weight")
b = ts.variable(rand, name: "bias")
pred = X * W + b
cost = ts.reduce_sum(ts.pow(pred - Y, 2)) / ( 2 * 10)
cost.to_math # "(reduce_sum(|((((Placeholder: * weight) + bias) - Placeholder_2:)^2)|) / 20.0)"
```

breakpoints can also be set, block will be evaluated during computation

```ruby
a = ts.constant([2,2])
b = ts.constant([3,3])

f = ts.matmul(a, b).breakpoint! { |tensor, a, b, result_value| binding.pry }

ts.session.run(f)
```

### OpenCL

For OpenCL support, make sure that the required OpenCL drivers for your hardware are correctly installed on your system.
Also OpenCL only supports ruby-mri at the moment.

Also include the following gem in your project:

```
gem 'opencl_ruby_ffi'
```

To use the opencl evaluator instead of the ruby evaluator simply add require it.

```ruby
require 'tensor_stream/evaluator/opencl/opencl_evaluator'
```

Adding the OpenCL evaluator should expose additional devices available to tensor_stream

```ruby
ts.list_local_devices
# ["job:localhost/ts:ruby:cpu", "job:localhost/ts:opencl:apple:0", "job:localhost/ts:opencl:apple:1"]
```
Here we see 1 "ruby" cpu device and 2 opencl "apple" devices (Intel CPU, Intel Iris GPU)

By default TensorStream will determine using the given evaluators the best possible
placement for each tensor operation

```ruby
require 'tensor_stream/evaluator/opencl/opencl_evaluator'

# set session to use the opencl evaluator
sess = ts.session

sess.run(....) # do stuff

```

You can manually place operations using ts.device e.g:

```ruby
ts = TensorStream
# Creates a graph. place in the first OpenCL CPU device

a, b = ts.device('/cpu:0') do
  a = ts.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [2, 3], name: 'a')
  b = ts.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [3, 2], name: 'b')
  [a, b]
end

c = ts.device('/device:GPU:0') do
  ts.matmul(a, b)
end

# Creates a session with log_device_placement set to True.
sess = ts.session(log_device_placement: true)
# Runs the op.
print(sess.run(c))

# a : apple:0
# b : apple:0
# a_1 : apple:0
# b_1 : apple:0
# matmul:0 : apple:1
# [[22.0, 28.0], [49.0, 64.0]] => nil
```

To force the ruby evaluator even with the OpenCL evaluator loaded you can use:

```ruby
ts.device('/ts:ruby:cpu') do
    # put ops here
end
```

Note that the OpenCL evaluator provides speedup if you are using large tensors, tensors that are only using scalars like the linear regression sample will actually be slower.

samples/nearest_neighbor.rb contains a sample that uses opencl.

# Visualization

tensorstream does not support tensorboard yet, but a graphml generator is included:

```ruby
tf = TensorStream
a = tf.constant(1.0)
b = tf.constant(2.0)
result = a + b
sess = tf.session
sess.run(result)

File.write('gradients.graphml', TensorStream::Graphml.new.get_string(result)) # dump graph only
File.write('gradients.graphml', TensorStream::Graphml.new.get_string(result, sess)) # dump with values from session
```

the resulting graphml is designed to work with yED, after loading the graph change layout to "Flowchart" for best results

## Exporting to TensorFlow

Still in alpha but tensorstream supports TensorFlows as_graph_def serialization method:

```ruby
tf = TensorStream
a = tf.constant(1.0)
b = tf.constant(2.0)
result = a + b
File.write("model.pbtext", result.graph.as_graph_def)
```

## Roadmap

- Docs
- Complete low-level op support
- SciRuby evaluator
- Opencl evaluator
- TensorFlow savemodel compatibility

## Issues

- This is an early preview release and many things still don't work
- Performance is not great, at least until the opencl and/or sciruby backends are complete
- However if you really need an op supported please feel free to file a pull request with the corresponding failing test (see spec/operation_spec.rb)

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/[USERNAME]/tensor_stream. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.


## License

The gem is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).

