require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Yaml do
  let(:tf) { TensorStream }
  let(:sess) { tf.session }
  let(:serializer) do
    TensorStream::Pbtext.new
  end

  before do
    @fixture = File.read(File.join('spec','fixtures', 'test.pbtxt.proto'))
    tf.reset_default_graph
  end

  xit "saves a graph into pbtext format" do
    # construct a graph
    a = tf.constant([[1.0, 1.1, 2.2, 3.3], [1.1, 2.2, 3.3, 4.0]])
    b = tf.constant(2.0)
    c = tf.variable(1.0, name: "v1")
    d = tf.constant([1,2,3,4])
    e = tf.constant("Hello World")
    f = tf.placeholder(:float32)
    g = tf.constant(["h","e","l","l","o"])
    func = a * b + c
    func2 = tf.reduce_sum(a, [0])

    tf.train.write_graph(tf.get_default_graph, '/tmp', 'ts_test_graph.yaml', serializer: described_class)
    expected_content = File.read(File.join('spec', 'fixtures', 'test.yaml'))
    test_content = File.read(File.join('/tmp', 'ts_test_graph.yaml'))
    expect(test_content).to eq(expected_content)
  end

  xcontext "save and restore a model" do
    specify do
      learning_rate = 0.01
      training_epochs = 2
      display_step = 50
      srand(1234)
      train_X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
      7.042,10.791,5.313,7.997,5.654,9.27,3.1]
      train_Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
      2.827,3.465,1.65,2.904,2.42,2.94,1.3]

      n_samples = train_X.size

      X = TensorStream.placeholder("float")
      Y = TensorStream.placeholder("float")

      # Set model weights
      W = TensorStream.variable(rand, name: "weight")
      b = TensorStream.variable(rand, name: "bias")

      # Construct a linear model
      pred = X * W + b

      # Mean squared error
      cost = TensorStream.reduce_sum(TensorStream.pow(pred - Y, 2.0)) / ( 2.0 * n_samples)

      optimizer = TensorStream::Train::GradientDescentOptimizer.new(learning_rate).minimize(cost)

      # Initialize the variables (i.e. assign their default value)
      init = TensorStream.global_variables_initializer
      sess.run(init)
      train_X.zip(train_Y).each do |x,y|
        sess.run(optimizer, feed_dict: {X => x, Y => y})
      end

      tf.train.write_graph(tf.get_default_graph, '/tmp', 'ts_test_graph_lg.yaml', serializer: described_class)
    end
  end
end