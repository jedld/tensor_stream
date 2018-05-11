# TensorStream

A reimplementation of TensorFlow for ruby. This is a ground up implementation with no dependency on TensorFlow. Effort has been made to make the programming style as near to TensorFlow as possible, comes with a pure ruby evaluator by default as well with support for an opencl evaluator.

The goal of this gem is to have a high performance machine learning and compute solution for ruby with support for a wide range of hardware and software configuration.

## Features

- Replicates most of the commonly used low-level tensorflow ops
- Supports auto-differentiation via tf.gradients (mostly)
- Provision to use your own opcode evaluator (opencl, sciruby and tensorflow backends planned)
- Goal is to be as close to TensorFlow in behavior but with some freedom to add ruby specific enhancements (with lots of test cases)

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
W = tf.Variable(rand, name: "weight")
b = tf.Variable(rand, name: "bias")

# Construct a linear model
pred = X * W + b

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / ( 2 * n_samples)

optimizer = TensorStream::Train::GradientDescentOptimizer.new(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

tf.Session do |sess|
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

## Roadmap

- Docs
- Complete low-level op support
- SciRuby evaluator
- Opencl evaluator
- TensorFlow savemodel compatibility

## Issues

- This is an early preview release and many things still don't work
- Performance is not great, at least until the opencl and/or sciruby backends are complete

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/[USERNAME]/tensor_stream. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.


## License

The gem is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).

