require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe "Performs a linear regression" do
  it "performs a linear regression" do
    learning_rate = 0.01
    training_epochs = 2
    display_step = 50

    train_X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
    7.042,10.791,5.313,7.997,5.654,9.27,3.1]
    train_Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
    2.827,3.465,1.65,2.904,2.42,2.94,1.3]

    n_samples = train_X.size

    X = TensorStream.placeholder("float")
    Y = TensorStream.placeholder("float")

    # Set model weights
    W = TensorStream.Variable(rand, name: "weight")
    b = TensorStream.Variable(rand, name: "bias")

    # Construct a linear model
    pred = X * W + b

    # Mean squared error
    cost = TensorStream.reduce_sum(TensorStream.pow(pred - Y, 2)) / ( 2 * n_samples)

    optimizer = TensorStream::Train::GradientDescentOptimizer.new(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = TensorStream.global_variables_initializer()

    TensorStream.Session do |sess|
      sess.run(init)
      (0..training_epochs).each do |epoch|
        train_X.zip(train_Y).each do |x,y|
          sess.run(optimizer, feed_dict: {X => x, Y => y})
        end

        if (epoch+1) % display_step == 0
          c = sess.run(cost, feed_dict: {X => train_X, Y => train_Y})
          puts("Epoch:", '%04d' % (epoch+1), "cost=",  c, \
              "W=", sess.run(W), "b=", sess.run(b))
        end
      end

      puts("Optimization Finished!")
      training_cost = sess.run(cost, feed_dict: { X => train_X, Y => train_Y})
      puts("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')  
    end
  end
end