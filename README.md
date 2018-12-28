[![Gem Version](https://badge.fury.io/rb/tensor_stream.svg)](https://badge.fury.io/rb/tensor_stream)[![CircleCI](https://circleci.com/gh/jedld/tensor_stream.svg?style=svg)](https://circleci.com/gh/jedld/tensor_stream) [![Join the chat at https://gitter.im/tensor_stream/Lobby](https://badges.gitter.im/tensor_stream/Lobby.svg)](https://gitter.im/tensor_stream/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# TensorStream

An opensource machine learning framework for ruby. Designed to run on a wide variety of ruby implementations (JRuby, TruffleRuby, MRI) as well as an option for High Performance computation (OpenCL).

This is a framework is heavily influenced by tensorflow and aims to be familiar with tensorflow users. This is a ground up implementation with no dependency on TensorFlow. Effort has been made to make the programming style as near to TensorFlow as possible, comes with a pure ruby evaluator by default with support for an opencl evaluator for large models and datasets.

## Goals & Features

- Easy to use - Improve model readability
- Replicates most of the commonly used low-level tensorflow ops (tf.add, tf.constant, tf.placeholder, tf.matmul, tf.sin etc...)
- Supports auto-differentiation using formal derivation
- Extensible - use your own opcode evaluator (OpenCL and Pure ruby currently supported)

## Compatibility

TensorStream comes with a pure ruby and OpenCL implementation out of the box. The pure ruby implementation
is known to work with most ruby implementations including TruffleRuby, JRuby as well as jit enabled versions of mri (ruby-2.6.0).

OpenCL is supported only on mri implementations of ruby. This can be enabled by adding OpenCL evaluator gem (Make sure you have OpenCL drivers installed correctly on your system):

```Gemfile
gem 'tensor_stream-opencl'
```

and then (without bundler)

```ruby
require 'tensor_stream/opencl'
```

OpenCL is basically a requirement for deep learning and image processing tasks as the ruby implementation is too slow even with jit speedups using latest ruby implementations.

OpenCL kernels used by tensorstream can be found at tensor_stream/lib/evaluator/opencl/kernels. These are non specific and should work with any device that supports OpenCL including intel GPUs and CPUs, as well as GPUs from Nvidia and AMD.

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

Usage is similar to how you would use TensorFlow except with ruby syntax.
There are also enhancements to the syntax to make it as consice as possible.

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

# X = tf.placeholder("float")
X = Float.placeholder

# Y = tf.placeholder("float")
Y = Float.placeholder

# Set model weights
# W = tf.variable(rand, name: "weight")
W = rand.t.var name: "weight"

# b = tf.variable(rand, name: "bias")
b = rand.t.var name: "bias"

# Construct a linear model
pred = X * W + b

# Mean squared error
cost = ((pred - Y) ** 2).reduce / ( 2 * n_samples)

# optimizer = TensorStream::Train::MomentumOptimizer.new(learning_rate, momentum, use_nesterov: true).minimize(cost)
# optimizer = TensorStream::Train::AdamOptimizer.new(learning_rate).minimize(cost)
# optimizer = TensorStream::Train::AdadeltaOptimizer.new(1.0).minimize(cost)
# optimizer = TensorStream::Train::AdagradOptimizer.new(0.01).minimize(cost)
# optimizer = TensorStream::Train::RMSPropOptimizer.new(0.01, centered: true).minimize(cost)
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

# Variables and Constants

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
c = ts.constant(1.0)

# concise way when initializing using a constant
w = 0.t.var name: 'weights'
c = 1.0.t
```

Calling .t to Integer, Array and Float types converts it into a tensor

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

# Another a bit more terse way
x = Float.placeholder shape: [1024, 1024]
y = Float.placeholder shape: [nil, 1024]
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

```Gemfile
gem 'tensor_stream-opencl'
```

To use the opencl evaluator instead of the ruby evaluator simply require it (if using rails this should be loaded automatically).

```ruby
require 'tensor_stream/opencl'
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
require 'tensor_stream/opencl'

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

## Export Import Models from tensorflow

Experimental support for parsing and exporting pbtext files are supported:

Exporting

```ruby
a = ts.constant([1.0, 1.0])
b = ts.constant([1.5, 1.5])
f = a + b

File.write('my_model.pbtext', f.graph.as_graph_def)
```

Importing (Experimental)

Note that not all tensorflow ops are supported, warnings will be showed
if a certain operation is not supported yet.


```ruby
  pbtext = File.read(File.join('linear_regression.pbtxt'))

  # create a graph from pbtext file
  graph = TensorStream::Graph.parse_from_string(pbtext)

  # reference a tensor by name from the created graph,
  # for example you have a tensor named out
  tensor = graph.get_tensor_by_name("out")

  # set graph as default and do operations on it
  graph.as_default do
    sess = ts.session
    expect(tr(sess.run(tensor))).to eq([[1.0, 1.0], [1.0, 1.0]])
  end

```

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

## Performance notes

Comparative performance with respect to other ruby libraries have not yet been performed. However it is
notable that TruffleRuby and ruby-2.6.0-preview2 with the --jit flag performs considerably better with respect
to previous versions of ruby(< 2.6)

Benchmarks running samples/linear_regression.rb on an Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz

ruby 2.4

```
$ ruby -v
ruby 2.4.0p0 (2016-12-24 revision 57164) [x86_64-linux]
$ ruby samples/linear_regression.rb
495 seconds 10000 epochs
```

ruby 2.6.0-preview2

```
$ ruby -v
ruby 2.6.0preview2 (2018-05-31 trunk 63539) [x86_64-linux]
$ ruby --jit samples/linear_regression.rb
394 seconds 10000 epochs
```

truffleruby
```
$ ruby -v
truffleruby 1.0.0-rc5, like ruby 2.4.4, GraalVM CE Native [x86_64-linux]
219 seconds 10000 epochs
```

For training large networks that works on images, the opencl evaluator is the only way to go.

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