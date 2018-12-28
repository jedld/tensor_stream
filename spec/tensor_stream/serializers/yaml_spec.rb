require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Yaml do
  let(:tf) { TensorStream }
  let(:sess) { tf.session }
  let(:serializer) do
    TensorStream::Pbtext.new
  end

  before do
    tf.reset_default_graph
  end

  it "saves a graph into yaml format" do
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

    tf.train.write_graph(tf.get_default_graph, '/tmp', 'ts_test_graph_simple.yaml', serializer: described_class)
    expected_content = File.read(File.join('spec', 'fixtures', 'ts_test_graph_simple.yaml'))
    test_content = File.read(File.join('/tmp', 'ts_test_graph_simple.yaml'))
    expect(test_content).to eq(expected_content)
  end

  context "neural net test save test" do
    def neural_net(x, weights, biases)
      tf = TensorStream
      # Hidden fully connected layer with 256 neurons
      layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
      # Hidden fully connected layer with 256 neurons
      layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
      # Output fully connected layer with a neuron for each class
      tf.matmul(layer_2, weights['out']) + biases['out']
    end

    specify do
      learning_rate = 0.001
      momentum = 0.01
      n_hidden_1 = 256 # 1st layer number of neurons
      n_hidden_2 = 256 # 2nd layer number of neurons
      num_input = 784 # MNIST data input (img shape: 28*28)
      num_classes = 10 # MNIST total classes (0-9 digits)

      # tf Graph input
      X = tf.placeholder(:float64, shape: [nil, num_input])
      Y = tf.placeholder(:float64, shape: [nil, num_classes])

      # Store layers weight & bias
      weights = {
          'h1' => tf.variable(tf.random_normal([num_input, n_hidden_1]), dtype: :float64, name: 'h1'),
          'h2' => tf.variable(tf.random_normal([n_hidden_1, n_hidden_2]), dtype: :float64, name: 'h2'),
          'out' => tf.variable(tf.random_normal([n_hidden_2, num_classes]), dtype: :float64, name: 'out')
      }

      biases = {
          'b1' => tf.variable(tf.random_normal([n_hidden_1]), dtype: :float64, name: 'b1'),
          'b2' => tf.variable(tf.random_normal([n_hidden_2]), dtype: :float64, name: 'b2'),
          'out' => tf.variable(tf.random_normal([num_classes]), dtype: :float64, name: 'out2')
      }

      # Construct model
      logits = neural_net(X, weights, biases)
      prediction = tf.nn.softmax(logits)

      # Define loss and optimizer
      loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
          logits: logits, labels: Y))

      optimizer = TensorStream::Train::MomentumOptimizer.new(learning_rate, momentum, use_nesterov: true)
      train_op = optimizer.minimize(loss_op)

      # Evaluate model
      correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_pred, :float32))

      # tf.add_check_numerics_ops

      # Initialize the variables (i.e. assign their default value)
      init = tf.global_variables_initializer

      actual_model = tf.train.write_graph(tf.get_default_graph, '/tmp', 'ts_test_graph_nn.yaml', serializer: described_class)
      expected_model = File.read(File.join('spec', 'fixtures', 'ts_test_graph_nn.yaml'))
      expect(actual_model).to eq(expected_model)
    end
  end

  context "save and restore a model" do
    specify "save" do
      learning_rate = 0.01
      training_epochs = 2
      display_step = 50
      srand(1234)
      train_X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
      7.042,10.791,5.313,7.997,5.654,9.27,3.1]
      train_Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
      2.827,3.465,1.65,2.904,2.42,2.94,1.3]

      n_samples = train_X.size

      X = TensorStream.placeholder("float", name: 'X')
      Y = TensorStream.placeholder("float", name: 'Y')

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

      expect(sess.run(pred, feed_dict: {X => train_X[0], Y => train_Y[0]})).to eq(1.3576718783214332)

      tf.train.write_graph(tf.get_default_graph, '/tmp', 'ts_test_graph_lg.yaml', serializer: described_class)
      saved_model = File.read(File.join('/tmp', 'ts_test_graph_lg.yaml'))
      expected_model = File.read(File.join('spec', 'fixtures', 'ts_test_graph_lg.yaml'))
      expect(saved_model).to eq(expected_model)
    end

    specify "restore" do
      graph = tf.get_default_graph
      expected_model = File.read(File.join('spec', 'fixtures', 'ts_test_graph_lg.yaml'))
      TensorStream::YamlLoader.new.load_from_string(expected_model)

      init = graph.get_tensor_by_name("/flow_group")
      optimizer = graph.get_tensor_by_name("GradientDescent/flow_group")
      pred = graph.get_tensor_by_name("add")
      sess.run(init)

      srand(1234)
      train_X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
      7.042,10.791,5.313,7.997,5.654,9.27,3.1]
      train_Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
      2.827,3.465,1.65,2.904,2.42,2.94,1.3]

      train_X.zip(train_Y).each do |x,y|
        sess.run(optimizer, feed_dict: {"X" => x, "Y" => y})
      end
      expect(sess.run(pred, feed_dict: {X => train_X[0], Y => train_Y[0]})).to eq(1.3576718783214332)
    end
  end
end