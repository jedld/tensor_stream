require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream do
  let(:tf) { TensorStream }

  before do
    TensorStream.disable_eager_execution
  end

  describe ".VERSION" do
    it "returns the version" do
      expect(TensorStream.version).to eq("0.1.4")
    end
  end

  describe ".enable_eager_execution" do
    it "enables eager execution" do
      TensorStream.enable_eager_execution
      expect(TensorStream.executing_eagerly?).to be
      a = TensorStream.constant(2)
      b = TensorStream.constant(3)
      print("a = %i" % a)
      print("b = %i" % b)

      x = [[2.0]]
      m = TensorStream.matmul(x, x)
      expect(m.to_a).to eq([[4.0]])

      d = TensorStream.constant(3.1)
      expect(d.to_f).to eq(3.1)
    end
  end

  describe ".trainable_variables" do
    it "Retrieves trainable variables for the current graph" do
      a = tf.variable(1, dtype: :float)
      b = tf.variable(2, dtype: :int32)
      c = tf.variable(2, dtype: :float32, trainable: false)

      expect(TensorStream.trainable_variables.map(&:name)).to eq([a, b].map(&:name))
    end
  end
end