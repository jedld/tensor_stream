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
      expect(described_class.ops.keys.size).to eq(92)
    end

    specify do
      expect(described_class.ops.keys.sort).to eq(%i[
        abs
        add
        apply_gradient_descent
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
        dynamic_stitch
        equal
        exp
        eye
        fill
        floor
        floor_div
        floor_mod
        flow_dynamic_stitch
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
        maximum
        mean
        mod
        mul
        neg
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
        range
        rank
        real_div
        reciprocal
        reshape
        restore_v2
        round
        save_v2
        sec
        select
        shape
        sigmoid
        sigmoid_grad
        sign
        sin
        size
        slice
        softmax
        softmax_cross_entropy_with_logits_v2
        softmax_cross_entropy_with_logits_v2_grad
        softmax_grad
        sqrt
        square
        squared_difference
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
end