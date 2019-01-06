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
  it_behaves_like "images ops"
  it_behaves_like "TensorStream::Train::Saver"
  it_behaves_like "optimizer evaluator"
  it_behaves_like "standard nn ops evaluator"
  it_behaves_like "freezer ops"

  context "supported ops" do
    specify do
      expect(described_class.ops.keys.size).to eq(129)
    end

    specify do
      expect(described_class.ops.keys.sort).to eq(%i[
        abs
        acos
        add
        add_n
        apply_adadelta
        apply_adagrad
        apply_adam
        apply_centered_rms_prop
        apply_gradient_descent
        apply_momentum
        apply_rms_prop
        arg_max
        arg_min
        argmax
        argmin
        asin
        assert_equal
        assign
        assign_add
        assign_sub
        atan
        broadcast_gradient_args
        broadcast_transform
        case
        case_grad
        cast
        ceil
        check_numerics
        concat
        concat_v2
        const
        conv2d
        conv2d_backprop_filter
        conv2d_backprop_input
        cos
        cumprod
        decode_png
        div
        dynamic_stitch
        encode_png
        equal
        exp
        expand_dims
        eye
        fill
        floor
        floor_div
        floor_mod
        flow_dynamic_stitch
        flow_group
        gather
        glorot_uniform
        greater
        greater_equal
        identity
        index
        invert_permutation
        less
        less_equal
        log
        log1p
        log_softmax
        logical_and
        mat_mul
        max
        maximum
        mean
        min
        minimum
        mod
        mul
        neg
        negate
        no_op
        not_equal
        ones
        ones_like
        pad
        placeholder
        pow
        print
        prod
        random_standard_normal
        random_uniform
        range
        rank
        real_div
        reciprocal
        relu6
        reshape
        restore_ts
        round
        save_ts
        select
        setdiff1d
        shape
        shape_n
        sigmoid
        sigmoid_grad
        sign
        sin
        size
        slice
        softmax
        softmax_cross_entropy_with_logits
        softmax_cross_entropy_with_logits_v2
        softmax_grad
        sparse_softmax_cross_entropy_with_logits
        split
        sqrt
        square
        squared_difference
        squeeze
        stack
        stop_gradient
        sub
        sum
        tan
        tanh
        tanh_grad
        tile
        transpose
        truncate
        truncated_normal
        unstack
        variable_v2
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