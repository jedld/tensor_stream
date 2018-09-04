require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::MathGradients do
  let(:tf) { TensorStream }
  let(:sess) { tf.session(:ruby_evaluator) }

  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    tf.reset_default_graph
    sess.clear_session_cache
  end

  context "addition" do
    it "handles shape differences, rank 2 vs 1" do
      a = tf.constant([[1, 2],[3, 4],[5, 6]])
      b = tf.constant([1, 1])
      sum = a + b
      g = tf.gradients(sum, [a, b])

      expect(sess.run(g)).to eq([[[1, 1], [1, 1], [1, 1]], [3, 3]])
    end

    it "handles shape differences, rank 2 vs 0" do
      a = tf.constant([[1, 2],[3, 4],[5, 6]])
      b = tf.constant(1)
      sum = a + b
      g = tf.gradients(sum, [a, b])

      expect(sess.run(g)).to eq([[[1, 1], [1, 1], [1, 1]], 6])

      sum2 = b + a
      g2 = tf.gradients(sum2, [a, b])
      expect(sess.run(g2)).to eq([[[1, 1], [1, 1], [1, 1]], 6])
    end
  end

  context "subtraction" do
    it "handles shape differences, rank 2 vs 1" do
      a = tf.constant([[1, 2],[3, 4],[5, 6]])
      b = tf.constant([1, 1])
      sum = a - b
      g = tf.gradients(sum, [a, b])

      expect(sess.run(g)).to eq([[[1, 1], [1, 1], [1, 1]], [-3, -3]])
    end

    it "handles shape differences, rank 2 vs 0" do
      a = tf.constant([[1, 2],[3, 4],[5, 6]])
      b = tf.constant(1)
      sum = a - b
      g = tf.gradients(sum, [a, b])

      expect(sess.run(g)).to eq([[[1, 1], [1, 1], [1, 1]], -6])
    end
  end

  it "computes for the derivative of a matrix multiplication operation" do
    y = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype: :float32)
    x = tf.constant([[4.0, 5.0], [5.0, 6.0]], dtype: :float32)

    c = tf.matmul(x, y)

    expect(sess.run(c)).to eq([[19, 28], [23, 34]])
    c_grad = tf.gradients(c, [x, y])
    expect(sess.run(c_grad)).to eq([
      [[3.0, 7.0], [3.0, 7.0]],
      [[9.0, 9.0], [11.0, 11.0]]
    ])
  end

  it 'should properly handle the gradient of non cubic matrices' do
    y = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype: :float32)
    z = tf.constant([[4.0, 5.0]], dtype: :float32)
    cz = tf.matmul(z, y)
    z_grad = tf.gradients(cz, [y])
    expect(sess.run(z_grad)).to eq([
      [[4.0, 4.0], [5.0, 5.0]]
    ])
  end

  it 'should handle matrix gradients with incompatible transpositions' do
    y = tf.constant([[1.0, 2.0 , 2.1, 0.8], [3.0, 4.0, 3.1, 0.9]], dtype: :float32)
    z = tf.constant([[4.0, 5.0], [1.1, 3.2], [5.0, 3.1], [1.0, 1.0]], dtype: :float32)
    cz = tf.matmul(y, z)
    expect(tr(sess.run(cz))).to eq([[17.5, 18.71], [32.8, 38.31]])
    z_grad = tf.gradients(cz, [y, z])
    expect(tr(sess.run(z_grad))).to eq(
      [[[9.0 , 4.3, 8.1, 2.0 ],
        [9.0 , 4.3, 8.1, 2.0 ]],

        [
          [4.0 , 4.0 ],
          [6.0 , 6.0 ],
          [5.2, 5.2 ],
          [1.7, 1.7 ]]])
  end

  context "handle broadcasted args" do
    it "automatically reduces broadcasted args (axis = 0)" do
      a = tf.constant([
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [0.1, 2.0, 1.1, 4.0, 5.0],
          [1.0, 2.0, 3.0, 4.0, 5.0],
        ])

      b  = tf.constant([0.1, 0.2, 0.1, 0.5, 0.4])

      f = a * b
      expect(tr(f.eval)).to eq([
        [0.1, 0.4, 0.3, 2.0, 2.0],
        [0.01, 0.4, 0.11, 2.0, 2.0],
        [0.1, 0.4, 0.3, 2.0, 2.0]
      ])

      g = tf.gradients(f, [b])

      expect(sess.run(g).first).to eq([2.1, 6.0, 7.1, 12.0, 15.0])
    end

    it "sum automatically reduces broadcasted args (axis = 1)" do
      a = tf.constant([
        [1.0, 2.0],
        [0.4, 4.1],
        [0.2, 4.2],
      ])

      b = tf.constant([
        [1.0],
        [0.4],
        [0.1],
      ])

      f = a * b

      expect(tr(sess.run(f))).to eq(
        [[1.0, 2.0], [0.16, 1.64], [0.02, 0.42]]
      )

      ab_sum = a + b

      g = tf.gradients(ab_sum, [b])
      result = sess.run(g)
      expect(result).to eq([[[2.0],[2.0],[2.0]]])
    end

    it "chain rule automatically reduces broadcasted args (axis = 1)" do
      a = tf.constant([
        [1.0, 2.0],
        [0.4, 4.1],
        [0.2, 4.2],
      ])

      b = tf.constant([
        [1.0],
        [0.4],
        [0.1],
      ])

      ab_sum_sin = tf.sin(a + b)
      g2 = tf.gradients(ab_sum_sin, [b])
      result = sess.run(g2)
      expect(tr(result)).to eq([[[-1.4061], [0.4859], [0.5545]]])
    end

    xspecify "when columns don't match" do
      a = tf.constant([
        [1.0, 2.0, 0.3],
        [0.4, 4.1, 0.1],
        [0.2, 4.2, 0.1],
      ])

      b = tf.constant([
        [1.0, 0.8],
        [0.4, 0.2],
        [0.1, 0.1],
      ])

      f = a * b

      expect {
        f.eval
      }.to raise_error TensorStream::Evaluator::EvaluatorExcecutionException
    end
  end

  context "multivariate chain rule (scalar)" do
    it "supports chains of functions" do
      a = tf.constant(1.0)
      b = tf.constant(2.1)
      y = tf.pow(a, 2) + b
      z = tf.sin(y)
      g = tf.gradients(z, [a, b])
      expect(tr(sess.run(g))).to eq([-1.9983, -0.9991])
    end
  end

  context "placeholders" do
    let(:test_inputs) {
      [
        [0.5937, 0.2343, 1.4332, 0.4395],
        [-1.0227, -0.6915, 1.2367, 0.3452],
        [-0.5675, 1.0374, 1.0429, 0.8839],
        [-0.1066, -0.0469, -1.6317, -1.4836],
        [0.7835, -3.0105, 1.713, -0.4536],
        [-0.3076, 1.3662, -0.6537, 0.0905],
        [-0.2459, 0.2243, -2.7048, 0.848],
      ]
    }

    it "should handle placeholders" do
      x = tf.placeholder("float", shape: [nil, 4])
      y = tf.placeholder("float", shape: [nil, 2])
      cz = tf.matmul(x, y)
      z_grad = tf.gradients(cz, [x, y])
      expect(tr(sess.run(z_grad, feed_dict: {
        x => [[1.0, 2.0 , 2.1, 0.8], [3.0, 4.0, 3.1, 0.9]],
        y => [[4.0, 5.0], [1.1, 3.2], [5.0, 3.1], [1.0, 1.0]]}))).to eq([[[9.0, 4.3, 8.1, 2.0], [9.0, 4.3, 8.1, 2.0]], [[4.0, 4.0], [6.0, 6.0], [5.2, 5.2], [1.7, 1.7]]])
    end

    it "neural net gradients" do
      num_inputs = 4
      num_neurons = 5
      inputs = tf.placeholder("float", shape: [nil, num_inputs])
      biases = tf.constant([0.5012, 1.302, -1.6217, 0.669, 0.1494], name: 'b1')
      biases2 = tf.constant([0.2012, 1.102, -1.5217, 0.469, 0.0494], name: 'b2')

      weights = tf.constant([
        [-0.9135, 1.0376, 0.8537, 0.4376, 1.3255],
        [-0.5921, -1.4081, 1.0614, -0.5283, 1.1832],
        [0.7285, -0.7844, 0.1793, -0.5275, -0.4426],
        [-1.4976, 0.4433, 2.2317, -2.0479, 0.7791]], name: 'w')

      weights_layer2 = tf.constant([
        [-1.0465, -0.8766, 1.6849, -0.6625, 0.7928],
        [2.0412, 1.3564, 0.7905, 0.6434, -2.5495],
        [2.4276, -0.6893, -1.5917, 0.0911, 0.9112],
        [-0.012, 0.0794, 1.3829, -1.018, -0.9328],
        [0.061, 0.9791, -2.1727, -0.9553, -1.434]], name: 'w2')


      sess = tf.session(:ruby_evaluator)

      layer_1 =  tf.matmul(inputs, weights) + biases
      neural_net = tf.matmul(layer_1, weights_layer2) + biases2


      output = sess.run(neural_net, feed_dict: { inputs => test_inputs })

      expect(tr(output)).to eq([
        [2.2988, 2.3936, -4.4248, 0.7761, -1.6559],
        [-6.8807, -1.0869, 10.0429, 2.0307, 2.7878],
        [0.8557, -0.5106, -9.8451, 1.6428, 5.0675],
        [-10.8091, 5.7856, 18.6334, -4.0783, -11.8674],
        [-5.6659, 4.7026, 27.1012, 1.4605, -11.3158],
        [-0.6659, 3.0561, -6.1193, -1.0023, -2.2235],
        [9.2274, 9.8467, -7.1795, 2.2881, -13.3659]]
      )

      g = tf.gradients(neural_net, [weights, biases])
      g2 = tf.gradients(neural_net, [weights_layer2, biases2])

      weight_gradient, biases_gradient = sess.run(g, feed_dict: { inputs => test_inputs })
      weight_gradient2, biases_gradient2 = sess.run(g2, feed_dict: { inputs => test_inputs })
      expect(tr(weight_gradient)).to eq([
        [0.0942, -1.9924, -1.0031, 0.437, 3.075],
        [0.0957, -2.0234, -1.0187, 0.4438, 3.1229],
        [-0.047, 0.994, 0.5005, -0.218, -1.5341],
        [-0.0723, 1.5287, 0.7696, -0.3353, -2.3593]])

      expect(tr(biases_gradient)).to eq([-0.7553, 15.974, 8.0423, -3.5035, -24.6533])

      expect(tr(weight_gradient2)).to eq([
        [4.1451, 4.1451, 4.1451, 4.1451, 4.1451],
        [9.4119, 9.4119, 9.4119, 9.4119, 9.4119],
        [-11.4653, -11.4653, -11.4653, -11.4653, -11.4653],
        [3.1677, 3.1677, 3.1677, 3.1677, 3.1677],
        [-0.8315, -0.8315, -0.8315, -0.8315, -0.8315]
      ])

      expect(tr(biases_gradient2)).to eq([7.0, 7.0, 7.0, 7.0, 7.0])
     end

     it "computes for open ended shapes" do
      x = tf.constant([[1.0, 0.5, 4.0]])

      w = tf.constant([[0.4, 0.2],[0.1, 0.45],[0.2, 4.0]])

      w2 = tf.constant([[0.3, 0.2],[0.15, 0.45]])

      b= tf.constant([4.0, 5.0])
      b2= tf.constant([4.1, 5.1])

      matmul_layer_1 = tf.matmul(x, w)
      a = tf.sin(matmul_layer_1 + b)
      matmul_layer_2 = tf.matmul(a, w2)
      matmul_layer_2_add = matmul_layer_2 + b2

      a2 = tf.sin(matmul_layer_2_add)

      g2 = tf.gradients(a2, [ b], name: 'final')
      final_result = sess.run(g2)

      expect(tr(final_result)).to eq([[-0.0639, -0.0778]])
     end

     it "softmax with variables" do
        sess =tf.session([:ruby_evaluator])
        batch_x = [
          [0.686274, 0.10196, 0.6509, 1.0, 0.9686, 0.49803, 0.0, 0.0, 0.0, 0.0],
          [0.543244, 0.10123, 0.4509, 0.0, 0.6986, 0.39803, 1.0, 0.0, 0.0, 0.0]
        ]

        batch_y = [
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ]

        num_input = 10
        num_classes = 10
        n_hidden_1 = 4 # 1st layer number of neurons
        n_hidden_2 = 4 # 2nd layer number of neurons

        X = tf.placeholder(:float, shape: [nil, num_input])
        Y = tf.placeholder(:float, shape: [nil, num_classes])


        h1_init = tf.constant([[0.5937, 0.2343, 1.4332, 0.4395],
                  [-1.0227, -0.6915, 1.2367, 0.3452],
                  [-0.5675, 1.0374, 1.0429, 0.8839],
                  [-0.1066, -0.0469, -1.6317, -1.4836],
                  [0.7835, -3.0105, 1.713, -0.4536],
                  [-0.3076, 1.3662, -0.6537, 0.0905],
                  [-0.2459, 0.2243, -2.7048, 0.848],
                  [0.3589, 0.3542, -0.0959, -1.327],
                  [-0.4685, 0.0844, 0.2794, 2.1275],
                  [-1.0733, 0.6189, 0.845, 0.033]])

        h2_init = tf.constant([[0.5012, 1.302, -1.6217, 0.669], [0.1494, -0.7837, -0.2978, 1.7745], [1.9727, -0.5312, -0.7391, 0.9187], [-0.6412, -1.4434, -0.8801, 0.9343]])
        h3_init = tf.constant([[0.5012, 1.302, -1.6217, 0.669, 0.1494, -0.7837, -0.2978, 1.7745, 1.9727, -0.5312],
          [-0.7391, 0.9187, -0.6412, -1.4434, -0.8801, 0.9343, -0.1665, -0.0032, 0.2959, -2.0488],
          [-0.9135, 1.0376, 0.8537, 0.4376, 1.3255, -0.5921, -1.4081, 1.0614, -0.5283, 1.1832],
          [0.7285, -0.7844, 0.1793, -0.5275, -0.4426, -1.4976, 0.4433, 2.2317, -2.0479, 0.7791]])


        b1_init = tf.constant([0.1494, -0.7837, -0.2978, 1.7745])

        b2_init = tf.constant([1.9727, -0.5312, -0.7391, 0.9187])
        out_init = tf.constant([-0.6412, -1.4434, -0.8801, 0.9343, -0.1665, -0.0032, 0.2959, -2.0488, -0.9135, 1.0376])


        h1 = tf.variable(h1_init, dtype: :float, name: 'h1')
        h2 = tf.variable(h2_init, dtype: :float, name: 'h2')
        h3 = tf.variable(h3_init, dtype: :float, name: 'out')

        b1 = tf.variable(b1_init, dtype: :float, name: 'b1')
        b2 = tf.variable(b2_init, dtype: :float, name: 'b2')
        out = tf.variable(out_init, dtype: :float, name: 'out2')

        layer_1 = tf.add(tf.matmul(X, h1), b1)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, h2), b2)
        # Output fully connected layer with a neuron for each class
        logits = tf.matmul(layer_2, h3) + out
        prediction = tf.nn.softmax(logits)
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
          logits: logits, labels: Y))
        optimizer = TensorStream::Train::GradientDescentOptimizer.new(0.01)
        train_op = optimizer.minimize(loss_op)
        init = tf.global_variables_initializer
        sess.run(init)

        expect(tr(h1.read_value)).to eq(
          [[0.5937, 0.2343, 1.4332, 0.4395],
          [-1.0227, -0.6915, 1.2367, 0.3452],
          [-0.5675, 1.0374, 1.0429, 0.8839],
          [-0.1066, -0.0469, -1.6317, -1.4836],
          [0.7835, -3.0105, 1.713, -0.4536],
          [-0.3076, 1.3662, -0.6537, 0.0905],
          [-0.2459, 0.2243, -2.7048, 0.848],
          [0.3589, 0.3542, -0.0959, -1.327],
          [-0.4685, 0.0844, 0.2794, 2.1275],
          [-1.0733, 0.6189, 0.845, 0.033]])

        op = sess.run(train_op, feed_dict: { X => batch_x, Y => batch_y })

        expect(tr(h1.read_value)).to eq(
          [[0.599, 0.2335, 1.4156, 0.4438],
          [-1.0217, -0.6917, 1.2341, 0.3458],
          [-0.5632, 1.0368, 1.0262, 0.888],
          [-0.107, -0.0465, -1.6578, -1.4766],
          [0.7902, -3.0115, 1.6881, -0.4475],
          [-0.3037, 1.3656, -0.6665, 0.0936],
          [-0.2357, 0.2224, -2.7042, 0.847],
          [0.3589, 0.3542, -0.0959, -1.327],
          [-0.4685, 0.0844, 0.2794, 2.1275],
          [-1.0733, 0.6189, 0.845, 0.033]]) # wrong

        expect(tr(h2.read_value)).to eq([
          [0.4931, 1.3047, -1.6231, 0.6705],
          [0.1815, -0.7959, -0.2905, 1.7689],
          [1.9565, -0.5295, -0.7367, 0.9222],
          [-0.6531, -1.4317, -0.8927, 0.9353]]) # wrong


        expect(tr(h3.read_value)).to eq([
          [0.5019, 1.302, -1.6217, 0.6692, 0.1494, -0.7649, -0.3007, 1.7745, 1.954, -0.5293],
          [-0.7376, 0.9187, -0.6411, -1.4429, -0.8801, 0.9355, -0.1725, -0.0032, 0.2948, -2.045],
          [-0.912, 1.0376, 0.8538, 0.4381, 1.3255, -0.605, -1.414, 1.0614, -0.5154, 1.187],
          [0.7283, -0.7844, 0.1793, -0.5276, -0.4426, -1.5022, 0.4443, 2.2317, -2.0433, 0.7785]]) # correct

        expect(tr(b1.read_value)).to eq([0.1596, -0.7858, -0.3227, 1.7808]) # wrong
        expect(tr(b2.read_value)).to eq([1.9589, -0.525, -0.7437, 0.921]) # wrong
        expect(tr(out.read_value)).to eq([-0.6417, -1.4434, -0.8801, 0.9341, -0.1665, 0.0018, 0.298, -2.0488, -0.9185, 1.0363]) # correct

        op = sess.run(train_op, feed_dict: { X => batch_x, Y => batch_y })

        expect(tr(h1.read_value)).to eq( [[0.6032, 0.2326, 1.3984, 0.4484],
          [-1.0209, -0.6918, 1.2315, 0.3465],
          [-0.5597, 1.036, 1.0098, 0.8924],
          [-0.107, -0.0467, -1.6833, -1.4694],
          [0.7956, -3.0127, 1.6638, -0.441],
          [-0.3007, 1.3649, -0.679, 0.0969],
          [-0.228, 0.2209, -2.7038, 0.8464],
          [0.3589, 0.3542, -0.0959, -1.327],
          [-0.4685, 0.0844, 0.2794, 2.1275],
          [-1.0733, 0.6189, 0.845, 0.033]])
    end
  end
end