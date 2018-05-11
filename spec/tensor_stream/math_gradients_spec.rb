require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::MathGradients do
  let(:tf) { TensorStream }

  context "addition" do
    it "handles shape differences, rank 2 vs 1" do
      a = tf.constant([[1, 2],[3, 4],[5, 6]])
      b = tf.constant([1, 1])
      sum = a + b
      g = tf.gradients(sum, [a, b])

      expect(g.eval).to eq([[[1, 1], [1, 1], [1, 1]], [3, 3]])
    end

    it "handles shape differences, rank 2 vs 0" do
      a = tf.constant([[1, 2],[3, 4],[5, 6]])
      b = tf.constant(1)
      sum = a + b
      g = tf.gradients(sum, [a, b])

      expect(g.eval).to eq([[[1, 1], [1, 1], [1, 1]], 6])
    end
  end

  context "subtraction" do
    it "handles shape differences, rank 2 vs 1" do
      a = tf.constant([[1, 2],[3, 4],[5, 6]])
      b = tf.constant([1, 1])
      sum = a - b
      g = tf.gradients(sum, [a, b])

      expect(g.eval).to eq([[[1, 1], [1, 1], [1, 1]], [-3, -3]])
    end

    it "handles shape differences, rank 2 vs 0" do
      a = tf.constant([[1, 2],[3, 4],[5, 6]])
      b = tf.constant(1)
      sum = a - b
      g = tf.gradients(sum, [a, b])

      expect(g.eval).to eq([[[1, 1], [1, 1], [1, 1]], -6])
    end
  end

  it "computes for the derivative of a matrix multiplication operation" do
    y = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype: :float32)
    x = tf.constant([[4.0, 5.0], [5.0, 6.0]], dtype: :float32)

    c = tf.matmul(x, y)

    expect(c.eval).to eq([[19, 28], [23, 34]])
    c_grad = tf.gradients(c, [x, y])
    expect(c_grad.eval).to eq([
      [[3.0, 7.0], [3.0, 7.0]],
      [[9.0, 9.0], [11.0, 11.0]]
    ])
  end

  it 'should properly handle the gradient of non cubic matrices' do
    y = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype: :float32)
    z = tf.constant([[4.0, 5.0]], dtype: :float32)
    cz = tf.matmul(z, y)
    z_grad = tf.gradients(cz, [y])
    expect(z_grad.eval).to eq([
      [[4.0, 4.0], [5.0, 5.0]]
    ])
  end

  it 'should handle matrix gradients with incompatible transpositions' do
    y = tf.constant([[1.0, 2.0 , 2.1, 0.8], [3.0, 4.0, 3.1, 0.9]], dtype: :float32)
    z = tf.constant([[4.0, 5.0], [1.1, 3.2], [5.0, 3.1], [1.0, 1.0]], dtype: :float32)
    cz = tf.matmul(y, z)
    expect(tr(cz.eval)).to eq([[17.5, 18.71], [32.8, 38.31]])
    z_grad = tf.gradients(cz, [y, z])
    expect(tr(z_grad.eval)).to eq(
      [[[9.0 , 4.3, 8.1, 2.0 ],
        [9.0 , 4.3, 8.1, 2.0 ]], 

        [
          [4.0 , 4.0 ],
          [6.0 , 6.0 ],
          [5.2, 5.2 ],
          [1.7, 1.7 ]]])
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
      expect(tr(z_grad.eval(feed_dict: {
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


      sess = tf.Session()

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
  end
end