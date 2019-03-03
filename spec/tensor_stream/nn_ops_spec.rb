require "spec_helper"
require "benchmark"

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
end
