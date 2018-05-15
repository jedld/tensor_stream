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

  context "#shape_diff" do
    it "computes for the difference in shapes" do
      expect(instance.shape_diff([5],[])).to eq([5])
      expect(instance.shape_diff([5, 2],[])).to eq([5, 2])
      expect(instance.shape_diff([5, 2],[2])).to eq([5, 0])
      expect(instance.shape_diff([5, 4],[2, 1])).to eq([3, 3])
      expect(instance.shape_diff([5, 4],[5, 5])).to be_nil
      expect(instance.shape_diff([2, 2],[1])).to eq([2, 1])
    end
  end

  context "#broadcast" do
    context "gets compatible shapes for two tensors" do
      specify "scalar vs scalar" do
        expect(instance.broadcast(1.0, 1.0)).to eq([1.0, 1.0])
      end

      specify "1D vs constant" do
        expect(instance.broadcast([1.0, 2.0], 1.0)).to eq([[1.0, 2.0], [1.0, 1.0]])
        expect(instance.broadcast([1.0, 2.0, 1.0], 1.0)).to eq([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
      end

      specify "1D vs 1D" do
        expect(instance.broadcast([1.0, 2.0], 1.0)).to eq([[1.0, 2.0], [1.0, 1.0]])
        expect(instance.broadcast([1.0, 2.0, 3.0], [1.0])).to eq([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
      end

      specify "2D vs 1D" do
        expect(instance.broadcast([[1.0, 2.0], [1.0, 2.0]], 1.0)).to eq([[[1.0, 2.0], [1.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])
        expect(instance.broadcast([[1.0, 2.0], [1.0, 2.0]], [1.0])).to eq([[[1.0, 2.0], [1.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])
        expect(instance.broadcast([[1.0, 2.0], [1.0, 2.0]], [3.0, 3.1])).to eq([[[1.0, 2.0], [1.0, 2.0]], [[3.0, 3.1], [3.0, 3.1]]])
      end

      specify "2D vs 2D" do
        expect(instance.broadcast([[1.0, 2.0], [1.0, 2.0]], [[1.0], [1.0]])).to eq([[[1.0, 2.0], [1.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])
        expect(instance.broadcast([[1.0, 2.0, 1.1], [1.0, 2.0, 2.2]], [[1.0], [2.0]])).to eq( [[[1.0, 2.0, 1.1], [1.0, 2.0, 2.2]], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]])
      end
    end
  end

  context "#broadcast_dimensions" do
    it "can broadcast various tensors in various shapes" do
      a = [1.0]
      expect(instance.broadcast_dimensions(a, [5])).to eq([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
      expect(instance.broadcast_dimensions(a, [2, 1])).to eq([[1.0, 1.0], [1.0, 1.0]])
      expect(instance.broadcast_dimensions(a, [3, 1])).to eq([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

      a = [[1.0, 2.0]]
      b = [[1.0],[2.0]]
      expect(instance.broadcast_dimensions(a, [3, 0])).to eq([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
      expect(instance.broadcast_dimensions(b, [0, 1])).to eq([[1.0, 1.0], [2.0, 2.0]])
      expect(instance.broadcast_dimensions(a, [])).to eq([[1.0, 2.0]])
      expect(instance.broadcast_dimensions(b, [])).to eq([[1.0], [2.0]])
      expect(instance.broadcast_dimensions([1.0], [2, 1])).to eq([[1.0, 1.0], [1.0, 1.0]])
    end
  end

  context "private ops" do
    context ".get_broadcasted_array_args" do
      it "returns axis to be used for reduction a.rank > b.rank" do
        a = _op(:shape, tf.constant([[1,1],[1,1]]))
        b = _op(:shape, tf.constant([[1,1],[1,1]]))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([])

        a = _op(:shape, tf.constant(1))
        b = _op(:shape, tf.constant(1))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([])

        a = _op(:shape, tf.constant([1.0, 1.0]))
        b = _op(:shape, tf.constant(1))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([0])

        a = _op(:shape, tf.constant([[1.0, 1.0],[1.0,1.0]]))
        b = _op(:shape, tf.constant(1))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([1,0])

        a = _op(:shape, tf.constant([[1.0, 1.0],[1.0,1.0]]))
        b = _op(:shape, tf.constant([1.0, 1.0]))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([0])

        a = _op(:shape, tf.constant([[1.0, 1.0],[1.0,1.0]]))
        b = _op(:shape, tf.constant([[1.0], [1.0]]))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([1])

        a = tf.constant([[1.0, 1.0],[1.0,1.0]])
        b = tf.constant([[1.0, 1.0]])
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([0])
      end

      it "does nothing if a.rank < b.rank" do
        b = tf.constant([[1.0, 1.0],[1.0,1.0]])
        a = tf.constant([[1.0, 1.0]])
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([])

        b = _op(:shape, tf.constant([[1.0, 1.0], [1.0, 1.0]]))
        a = _op(:shape, tf.constant([[1.0], [1.0]]))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sb.eval).to eq([])
      end
    end
  end
end