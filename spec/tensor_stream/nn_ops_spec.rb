require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::NN do
  let(:tf) { TensorStream } # Tensorflow compatibility alias

  context ".softmax" do
    it "computes for the softmax of a group of values" do
      outputs = tf.constant([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]])
      expect(tr(tf.nn.softmax(outputs).eval)).to eq( [[0.0236, 0.0643, 0.1747, 0.4748, 0.0236, 0.0643, 0.1747], [0.0236, 0.0643, 0.1747, 0.4748, 0.0236, 0.0643, 0.1747]])
    end

    specify "gradients" do
      outputs = tf.constant([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
      f = tf.log(tf.nn.softmax(outputs))
      g = tf.gradients(f, [outputs])
      sess = tf.session
      expect(tr(sess.run(g))).to eq([[0.8345162, 0.5501684, -0.22276916, -2.323831, 0.8345162,
      0.5501684, -0.22276916]])
    end
  end

  context ".relu" do
    it "Computes rectified linear: max(features, 0)." do
      outputs = tf.constant([-1.0, -1.1, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
      f = tf.nn.relu(outputs)
      expect(f.eval).to eq([0, 0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
    end

    specify "gradients" do
      outputs = tf.constant([-1.0, -1.1, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
      f = tf.nn.relu(outputs)
      g = tf.gradients(f, [outputs])
      expect(g.eval).to eq([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    end
  end

  context ".sigmoid_cross_entropy_with_logits" do
    it "Measures the probability error in discrete classification tasks" do
      labels = tf.constant([[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]])
      outputs = tf.constant([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]])
      f = tf.nn.sigmoid_cross_entropy_with_logits(logits: outputs, labels: labels)
      expect(tr(f.eval)).to eq([[0.3133, 2.1269, 0.0486, 4.0181, 0.3133, 0.1269, 3.0486], [0.3133, 2.1269, 0.0486, 4.0181, 0.3133, 0.1269, 3.0486]])
    end

    specify "gradients" do
      labels = tf.constant([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
      outputs = tf.constant([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
      f = tf.nn.sigmoid_cross_entropy_with_logits(logits: outputs, labels: labels)
      g = tf.gradients(f, [labels, outputs])
      expect(tr(g.eval)).to eq([[-1.0, -2.0, -3.0, -4.0, -1.0, -2.0, -3.0], [-0.2689, 0.8808, -0.0474, 0.982, -0.2689, -0.1192, 0.9526]])
    end
  end

  context ".softmax_cross_entropy_with_logits" do
    it "Computes softmax cross entropy between logits and labels" do
      labels = tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
      outputs = tf.constant([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [1.1, 2.1, 3.0, 4.0, 1.0, 2.0, 3.0]])
      f = tf.nn.softmax_cross_entropy_with_logits(logits: outputs, labels: labels)
      expect(tr(f.eval)).to eq([3.7448, 2.654])
    end

    specify "gradients" do
      labels = tf.constant([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      outputs = tf.constant([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
      f = tf.nn.softmax_cross_entropy_with_logits(logits: outputs, labels: labels)
      g = tf.gradients(f, [outputs])
      expect(tr(g.eval)).to eq([[-0.9764, 0.0643, 0.1747, 0.4748, 0.0236, 0.0643, 0.1747]])
    end
  end
end