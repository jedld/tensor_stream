require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::NN do
  let(:tf) { TensorStream } # Tensorflow compatibility alias
  let(:sess) {TensorStream.session(:ruby_evaluator) }

  context ".relu" do
    it "Computes rectified linear: max(features, 0)." do
      outputs = tf.constant([-1.0, -1.1, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
      f = tf.nn.relu(outputs)
      expect(sess.run(f)).to eq([0, 0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
    end

    specify "gradients" do
      outputs = tf.constant([-1.0, -1.1, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
      f = tf.nn.relu(outputs)
      g = tf.gradients(f, [outputs])
      expect(sess.run(g)).to eq([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    end
  end

  context ".softmax_cross_entropy_with_logits" do
    it "Computes softmax cross entropy between logits and labels" do
      labels = tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
      outputs = tf.constant([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [1.1, 2.1, 3.0, 4.0, 1.0, 2.0, 3.0]])
      f = tf.nn.softmax_cross_entropy_with_logits_v2(logits: outputs, labels: labels)
      expect(tr(sess.run(f))).to eq([3.7448, 2.654])
    end

    specify "overflow resistance" do
      x = tf.constant([-2046.4904911315384, 2371.594564592362, -1920.025585664249, 266.06257844862205, 570.1462458227674, 2290.6715733914048, 1396.0319189271745, -2750.277642111798, 1758.5654697551304, 3116.9786057465503])
      label = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
      f = tf.nn.softmax_cross_entropy_with_logits_v2(logits: x, labels: label)
      expect(tr(sess.run(f))).to eq(1358.4131)
    end

    specify "gradients" do
      labels = tf.constant([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      outputs = tf.constant([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
      f = tf.nn.softmax_cross_entropy_with_logits_v2(logits: outputs, labels: labels)
      g = tf.gradients(f, [outputs])
      expect(tr(sess.run(g))).to eq([[-0.9764, 0.0643, 0.1747, 0.4748, 0.0236, 0.0643, 0.1747]])
    end

    specify "gradients overflow resistance" do
      x = tf.constant([-2046.4904911315384, 2371.594564592362, -1920.025585664249, 266.06257844862205, 570.1462458227674, 2290.6715733914048, 1396.0319189271745, -2750.277642111798, 1758.5654697551304, 3116.9786057465503])
      label = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
      f = tf.nn.softmax_cross_entropy_with_logits_v2(logits: x, labels: label)
      h = tf.nn.softmax(x)
      expect(sess.run(h)).to eq([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
      g = tf.gradients(f, [x])
      expect(tr(sess.run(g))).to eq([[ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  1.0]])
    end
  end
end