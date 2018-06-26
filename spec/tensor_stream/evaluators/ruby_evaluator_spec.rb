require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Evaluator::RubyEvaluator do
  let(:tf) { TensorStream }
  let(:sess) { TensorStream.session(:ruby_evaluator) }
  let(:execution_context) { Hash.new }
  let(:instance) { described_class.new(sess, {})}

  include TensorStream::OpHelper

  def create_session
    TensorStream.session
  end

  it_behaves_like "standard ops evaluator"
  it_behaves_like "TensorStream::Train::Saver"

  context "supported ops" do
    specify do
      expect(described_class.ops.keys.size).to eq(75)
    end

    specify do
      expect(described_class.ops.keys.sort).to eq(%i[
        abs
        add
        argmax
        argmin
        assign
        assign_add
        assign_sub
        broadcast_gradient_args
        broadcast_transform
        cast
        ceil
        check_numerics
        concat
        cond
        const
        cos
        div
        equal
        exp
        eye
        floor
        flow_group
        glorot_uniform
        greater
        greater_equal
        identity
        index
        less
        less_equal
        log
        log1p
        logical_and
        matmul
        max
        mean
        mul
        negate
        no_op
        not_equal
        ones
        ones_like
        pad
        pow
        print
        prod
        random_normal
        random_uniform
        rank
        reciprocal
        reduced_shape
        reshape
        round
        sec
        shape
        sigmoid
        sigmoid_grad
        sign
        sin
        slice
        softmax
        softmax_grad
        sqrt
        square
        stop_gradient
        sub
        sum
        tan
        tanh
        tanh_grad
        tile
        transpose
        truncate
        where
        zeros
        zeros_like])
    end
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

  context "private ops" do
    context ".reduced_shape" do
      it "returns the output shape of a tensor after reduction assuing keepdims= true" do
        input = tf.constant([[2,3],[3,4]])
        expect(sess.run(_op(:reduced_shape, tf.shape(input), 0))).to eq([1, 2])
      end
    end

    context ".get_broadcasted_array_args" do
      it "returns axis to be used for reduction a.rank > b.rank" do
        a = _op(:shape, tf.constant([[1,1],[1,1]]))
        b = _op(:shape, tf.constant([[1,1],[1,1]]))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sess.run(sb)).to eq([])

        a = _op(:shape, tf.constant(1))
        b = _op(:shape, tf.constant(1))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sess.run(sb)).to eq([])

        a = _op(:shape, tf.constant([1.0, 1.0]))
        b = _op(:shape, tf.constant(1))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sess.run(sb)).to eq([0])

        a = _op(:shape, tf.constant([[1.0, 1.0],[1.0,1.0]]))
        b = _op(:shape, tf.constant(1))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sess.run(sb)).to eq([1,0])

        a = _op(:shape, tf.constant([[1.0, 1.0],[1.0,1.0]]))
        b = _op(:shape, tf.constant([1.0, 1.0]))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sess.run(sb)).to eq([0])

        a = _op(:shape, tf.constant([[1.0, 1.0],[1.0,1.0]]))
        b = _op(:shape, tf.constant([[1.0], [1.0]]))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sess.run(sb)).to eq([1])

        a = tf.constant([[1.0, 1.0],[1.0,1.0]])
        b = tf.constant([[1.0, 1.0]])
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sess.run(sb)).to eq([0])
      end

      it "does nothing if a.rank < b.rank" do
        b = tf.constant([[1.0, 1.0],[1.0,1.0]])
        a = tf.constant([[1.0, 1.0]])
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sess.run(sb)).to eq([])

        b = _op(:shape, tf.constant([[1.0, 1.0], [1.0, 1.0]]))
        a = _op(:shape, tf.constant([[1.0], [1.0]]))
        sb = _op(:broadcast_gradient_args, a, b)
        expect(sess.run(sb)).to eq([])
      end
    end
  end
end