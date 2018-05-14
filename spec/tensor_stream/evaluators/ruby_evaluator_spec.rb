require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Evaluator::RubyEvaluator do
  let(:tf) { TensorStream }
  let(:session) { TensorStream.session }
  let(:execution_context) { Hash.new }
  let(:instance) { described_class.new(session, {})}

  include TensorStream::OpHelper

  it "can evaluate a tensor" do
    c = tf.constant(1.0)
    expect(instance.run(c, execution_context)).to eq(1.0)
  end

  it "can evaluate an array of tensors" do
    a = tf.constant(1.0)
    input = [tf.constant([1.0, 1.0]), tf.sin(a)]
    expect(instance.run(input, execution_context)).to eq([[1.0, 1.0], 0.8414709848078965])
  end

  context "#broadcast_dimensions" do
    it "can broadcast various tensors in various shapes" do
      a = 1.0
      expect(instance.broadcast_dimensions(a, [5])).to eq([1.0, 1.0, 1.0, 1.0, 1.0])
      expect(instance.broadcast_dimensions(a, [2, 2])).to eq([[1.0, 1.0], [1.0, 1.0]])
      expect(instance.broadcast_dimensions(a, [3, 2])).to eq([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

      a = [[1.0, 2.0]]
      b = [[1.0],[2.0]]
      expect(instance.broadcast_dimensions(a, [3, 0])).to eq([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
      expect(instance.broadcast_dimensions(b, [0, 1])).to eq([[1.0, 1.0], [2.0, 2.0]])
      expect(instance.broadcast_dimensions(a, [])).to eq([[1.0, 2.0]])
      expect(instance.broadcast_dimensions(b, [])).to eq([[1.0], [2.0]])
    end
  end

  context "private ops" do
    context ".get_broadcasted_array_args" do
      it "returns axis to be used for reduction a.rank > b.rank" do
        a = op(:shape, tf.constant([[1,1],[1,1]]))
        b = op(:shape, tf.constant([[1,1],[1,1]]))
        sb = op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([])

        a = op(:shape, tf.constant(1))
        b = op(:shape, tf.constant(1))
        sb = op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([])

        a = op(:shape, tf.constant([1.0, 1.0]))
        b = op(:shape, tf.constant(1))
        sb = op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([0])

        a = op(:shape, tf.constant([[1.0, 1.0],[1.0,1.0]]))
        b = op(:shape, tf.constant(1))
        sb = op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([1,0])

        a = op(:shape, tf.constant([[1.0, 1.0],[1.0,1.0]]))
        b = op(:shape, tf.constant([1.0, 1.0]))
        sb = op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([0])

        a = op(:shape, tf.constant([[1.0, 1.0],[1.0,1.0]]))
        b = op(:shape, tf.constant([[1.0], [1.0]]))
        sb = op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([1])

        a = tf.constant([[1.0, 1.0],[1.0,1.0]])
        b = tf.constant([[1.0, 1.0]])
        sb = op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([0])
      end

      it "does nothing if a.rank < b.rank" do
        b = tf.constant([[1.0, 1.0],[1.0,1.0]])
        a = tf.constant([[1.0, 1.0]])
        sb = op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([])

        b = op(:shape, tf.constant([[1.0, 1.0], [1.0, 1.0]]))
        a = op(:shape, tf.constant([[1.0], [1.0]]))
        sb = op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([])
      end
    end
  end
end