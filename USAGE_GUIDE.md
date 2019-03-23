Introduction
============

This document describes the basic usage of TensorStream and serves as a walthrough with regards to its features, limitations as well as various tools for debugging and development.

Since TensorStream is heavily based on TensorFlow, if you have used TensorFlow before then aside from the syntax a lot of things will be quite familiar.

What is TensorStream?
--------------------

TensorStream is an opensource framework for machine learning for ruby, its goal is to allow machine learning models to be easily built and run them in various hardware like GPUs and CPUs. It is heavily based on TensorFlow with the goal of being able to easily port its higher level libraries and model examples. As such it is also based on data flow graphs wherein you define computations and data flows between those computations in order to achieve the desired output.

TensorStream is designed to support various backends with a Pure Ruby and OpenCL implementation. These implementations are designed to work together, you can perform training on an OpenCL implementation (where you have a GPU) and then run the resulting trained model on a Pure Ruby implementation where you can deploy anywhere that you can run ruby on. TensorStream has been tested to run on most ruby implementations like MRI, JRuby and TruffleRuby.

Introduction to Tensors
-----------------------

Tensors are just a mathematical term to describe scalar, single and multidimensional arrays in a consistent manner. Though there is a formal mathematical definition for it, for all intents and purposes these are how data like numbers, strings are represented and structured in order to be fed into operations which in turn process them to be changed into another number or have its structure changed.

Tensors have properties that describe their shape and rank as well as their data type.

Below are examples of Constant Tensors. Their values are immutable and cannot be changed.

```ruby
t1 = 1.0.t                      # scalar a tensor or rank 0
t2 = [1.0, 2.1].t               # a float tensor of rank 1
t3 = [[2.0, 2.1], [2.1, 2.2]].t # a float tensor of rank 2
t4 = [[2, 2], [3, 3]].t         # an integer tensor of rank 2

# alternatively you can use tensorflow style constant definition
ts = TensorStream
t1 = ts.constant(1.0)
```

Notice that you can create a constant tensor by calling the .t method on an Integer, Float and an Array. You can also use the TensorStream.constant method to achieve the same effect.

tensors can be referenced later by giving it a name (They automatically get a name if you don't give it one)

```ruby
t1 = 1.0.t(name: 'c1')
t2 = [5.0].t
t2.name
=> "Const"

# tensorflow style
ts = TensorStream
t1 = ts.constant(1.0, name: 'c1')

# Reference later
graph = ts.get_default_graph
tx = graph['c1']
tx.run
=> 1.0
```

Tensor Shapes
-------------

The shape to use depends on what the data represents and what computation you want to achieve.

The shape of a tensor describes its structure or describes the dimensions of the array. So for example in order to represent a 28x28 2D grayscale image you would need a tensor with shape [28, 28] with each cell representing a single channel. If instead you have an 28x28 RGB image you would then need a tensor with shape [28, 28, 3], now you need 3 values to represent each pixel. Now what if you need to represent 100 RGB images? then it follows that you need a tensor of size [100, 28, 28, 3]!

Computations
------------

Naturally the whole point of all of this is to be able to perform computations. TensorStream supports all of the basic math operations you would expect, only beefed up to work with tensors:

```ruby
t1 = 1.0.t
t2 = 2.0.t
sum = t1 + t2
=> Op(add name: add shape: TensorShape([]) data_type: float32)
```

Note that sum did not actually compute the "sum" ... yet.
what happened is that you only defined the data flow graph, in order to get the actual result you need to run it in a session


```ruby
t1 = 1.0.t
t2 = 2.0.t
sum = t1 + t2
=> Op(add name: add shape: TensorShape([]) data_type: float32)

sess = TensorStream.session
sess.run(sum)
=> 3.0
sess.run(t1, t2) # pass multiple tensors/ops
=> [1.0, 2.0]

# this also works as a shortcut and is equivalent to above
sum.run
=> 3.0
```

TensorStream and TensorFlow (in non eager execution mode) works like this since it uses the dataflow graph to be able to perform gradient computation for machine learning operations. It also uses the graph structure to be able to run that computation in an optimal manner on various hardware like the GPU.

Of course operations on multidimensional arrays work as you would expect

```ruby
t1 = [1.0, 1.5].t
t2 = [1.2, 1.5].t
sum = t1 + t2
sum.run
=> [2.2, 3.0].t
```

There are a wealth of other operations available like reduction for example:

```ruby
t1 = [1.0, 1.5, 2.0].t
t1.reduce(:+).run
=> 4.5

# or tensorflow style
# ts.reduce_sum(t1)
```

Broadcast Operations
--------------------

So Tensor sizes don't have to be the same, you can, in some instances use a different but compatible size in order to perform an operation like below

```ruby
t1 = [[1.0, 1.5], [1.0, 1.5]].t
sum = t1 + 1.0
sum.run
=>  [[2.0, 2.5], [2.0, 2.5]].t
```

Here we "broadcasted" a scalar float constant to all cells in a tensor. If these were run on a GPU you can imagine that this operation can be run in parallel.

Below is another way, but using a tensor of rank 1 on a rank 2 tensor:

```ruby
t1 = [[1.0, 1.5], [1.0, 1.5]].t
sum = t1 + [1.0, 0.5].t
sum.run
=>  [[2.0, 2.0], [2.0, 2.0]].t
```

In this case we saw that a row by row operation was done instead. There are a number of operations that support broadcasting like multipliation, subtraction, divison etc.

Placeholders and Variables
--------------------------

There are special types of tensors that are frequently used in building a model in order
to serve as (Placeholders) for values as well as to store data that can be used in
succeeding sessions (Variables)

Placeholders are like parameters which take on a value during the time that the model is ran.

For example:

```ruby
param1 = Float.placeholder

sum = 2.0.t + param1
sess = TensorStream.session
sess.run(sum, feed_dict: { param1 => 1.0 })
=> 3.0
sess.run(sum, feed_dict: { param1 => 2.0 })
=> 4.0
```

Note that NOT passing a value for the placeholder will result in an error.

Variables on the other hand provide persistent data that survives between sessions, however they need to be initialized first otherwise an error will occur.

See below for an example:

```ruby
ts = TensorStream
var1 = 1.0.t.var(name: 'var')

acc = var1 + 1
assign = var1.assign(acc)

# Or tensorflow style
# var1 = ts.variable(1.0, dtype: :float32, name: 'var1')

# initialize the variables to their initial value
sess = TensorStream.session
init = ts.global_variables_initializer
sess.run(init)

# first run
sess.run(acc, assign)
=> [2.0, 2.0]
sess.run(acc, assign)
=> [3.0, 3.0]
```

Variables can be trainable or non-trainable. This property is used by training algorithms to determine if these will be updated during training.

```ruby
v = TensorStream.variable(1.0, name: 'v', trainable: false)
```

Graphs
------

Graphs hold the entire model data structure, each operation defined is stored in a graph which is later used during runtime to perform operations as well as during serialization and deserialization.

When there is no graph present when a tensor is defined, one will automatically be created and will serve as the "default" graph.

Access to the graph can be accomplished using the get_default_graph method.

```ruby
ts = TensorStream
graph = ts.get_default_graph

# access nodes
graph.nodes
=> {"Const"=>Op(const name: Const shape: TensorShape([]) data_type: float32), "Const_1"=>Op(const name: Const_1 shape: TensorShape([]) data_type: float32)}

```

The graph object can also be used to access collections like a list of variables

```ruby
vars = graph.get_collection(TensorStream::GraphKeys::GLOBAL_VARIABLES)
=> [Variable(Variable:0 shape: TensorShape([]) data_type: float32)]
```

High Performance Computing
--------------------------

TensorStream has been designed from the ground up to support multiple execution backends.

What this means is you can build your models once and then be able to execute them later on specialized hardware when available like GPUs.

An OpenCL backend is available that you can use for compute intensive taks like machine learning, especially those that use convolutional networks.

Using OpenCL is as simple as installing the tensorstream-opencl gem

```
gem install tensor_stream-opencl
```

You can then require the library in your programs and it will get used automatically (assuming you also installed OpenCL drivers for your system)

```ruby
require 'tensor_stream'

# enable OpenCL
require 'tensor_stream/opencl'

tf = TensorStream

srand(5)
seed = 5
tf.set_random_seed(seed)

SHAPES = [32, 32]
tf = TensorStream
sess = tf.session
large_tensor = tf.constant(sess.run(tf.random_uniform([256, 256])))

sum_axis_1 = tf.reduce_sum(large_tensor, 1)
sess.run(sum_axis_1)
```

Using OpenCL can improve performance dramatically in scenarios involving large tensors:

```
Linux 4.15.0-46-generic #49-Ubuntu SMP
model name	: AMD Ryzen 3 1300X Quad-Core Processor
OpenCL device NVIDIA CUDA GeForce GTX 1060 6GB
ruby 2.6.2p47 (2019-03-13 revision 67232) [x86_64-linux]

                                           user     system      total        real
pure ruby softmax        :             0.024724   0.000000   0.024724 (  0.024731)
opencl    softmax        :             0.006237   0.003945   0.010182 (  0.009005)
pure ruby matmul         :             0.679538   0.000000   0.679538 (  0.680048)
opencl    matmul         :             0.003456   0.007965   0.011421 (  0.008568)
pure ruby sum            :             3.210619   0.000000   3.210619 (  3.210064)
opencl sum               :             0.002431   0.008030   0.010461 (  0.007522)
pure ruby sum axis 1     :             3.208789   0.000000   3.208789 (  3.208125)
opencl sum axis 1        :             0.006075   0.003963   0.010038 (  0.007679)
pure ruby conv2d_backprop      :       3.738167   0.000000   3.738167 (  3.737946)
opencl conv2d_backprop         :       0.031267   0.003958   0.035225 (  0.030381)
pure ruby conv2d      :                0.794182   0.000000   0.794182 (  0.794100)
opencl conv2d         :                0.015865   0.004020   0.019885 (  0.016878)
```

A quick glance shows not a marginal increase but an order of magnitude performance increase in most operations.
In fact we are looking at almost a 200x faster compute on operations like matmul and softmax (essential operations in machine learning). This is not a surprise because of the "embarrasingly" parallel nature of machine learning computation. Because of this, GPUs are basically a requirement in most machine learning tasks.

The code containing these benchmarks can be found at:

tensor_stream-opencl/benchmark/benchmark.rb

Limitations
-----------

- Current version only supports dense tensors, meaning for multidimensional arrays each row in a dimension must have the same size. Examples below are not support as of now:

```ruby
[[1, 2], [1], [2, 3]] # second array has a different size than the others
```

- The ruby evaluator uses the ruby Float and Integer objects during computation as such the width of the data types (float32 vs float64, int32 vs int64) aren't really used. This however matters with the OpenCL evaluator that uses the width for determining the correct C data type to use in the OpenCL kernels.