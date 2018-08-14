require "spec_helper"
require 'benchmark'
require 'tensor_stream/evaluator/opencl/opencl_evaluator'

RSpec.describe TensorStream::Evaluator::OpenclEvaluator do
  let(:tf) { TensorStream }
  let(:sess) { TensorStream.session([:opencl_evaluator, :ruby_evaluator]) }
  let(:instance) { described_class.new(sess, TensorStream::Evaluator::OpenclEvaluator.default_device, {})}

  it_behaves_like "standard ops evaluator"
  it_behaves_like "TensorStream::Train::Saver"

  def create_session
    TensorStream.session([:opencl_evaluator, :ruby_evaluator])
  end

  context ".query_device" do
    it "selects the first GPU if there is one" do
      device = described_class.query_device("/device:GPU:0")
      expect(device).to be
      expect(device.type).to eq(:gpu)
      device = described_class.query_device("/gpu:0")
      expect(device).to be
      expect(device.type).to eq(:gpu)
    end

    context "tensor_stream convention" do
      it "selects a specific device on evaluator" do
      devices = tf.list_local_devices.select { |d| d =~ /opencl/ }
      device = described_class.query_device(devices.first)
      expect(device).to be
      end
    end
  end

  context "device placement test" do
    it "should evaluate tensors in appropriate device" do
      sess = TensorStream.session([:opencl_evaluator, :ruby_evaluator], log_device_placement: true)
      c = tf.device("/cpu:0") do
        a = tf.constant(1.0)
        b = tf.constant(1.0)
        a + b
      end
      expect(c.device).to eq("/cpu:0")
      d = tf.device("/device:GPU:0") do
        a = tf.constant(1.0)
        b = tf.constant(1.0)
        a + b
      end
      expect(d.device).to eq("/device:GPU:0")
      sess.run(c, d)
    end
  end

  context "supported ops" do
    specify do
      expect(described_class.ops.keys.size).to eq(77)
    end

    specify do
      expect(described_class.ops.keys.sort).to eq(%i[
        abs
        acos
        add
        add_n
        apply_gradient_descent
        argmax
        argmin
        asin
        assign
        assign_add
        assign_sub
        broadcast_gradient_args
        broadcast_transform
        cast
        ceil
        check_numerics
        cond
        cos
        div
        equal
        exp
        expand_dims
        fill
        floor
        floor_div
        floor_mod
        flow_group
        greater
        greater_equal
        identity
        index
        less
        less_equal
        log
        log1p
        log_softmax
        logical_and
        mat_mul
        max
        mean
        min
        mod
        mul
        negate
        no_op
        not_equal
        pow
        print
        prod
        rank
        real_div
        reciprocal
        reshape
        round
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
        stack
        stop_gradient
        sub
        sum
        tan
        tanh
        tanh_grad
        transpose
        where
      ])
    end

    it "allows automatic fallback" do
      a = tf.constant([1,2,3,4], dtype: :float32)
      c = tf.concat(a, 0)
      d = tf.sin(c)
      expect(tr(sess.run(d))).to eq(-0.544)
    end
  end

  context ".list_local_devices" do
    specify do
      expect(tf.list_local_devices.size > 1).to be
    end
  end

  describe "data types" do
    %i[int32 int16 int].each do |dtype|
      context "#{dtype}" do
        specify do
          a = tf.constant([1,2,3,4,5], dtype: dtype)
          b = tf.constant([5,6,7,8,9], dtype: dtype)
          f = a + b
          g = a * b
          h = a / b
          j = a - b

          expect(sess.run(f, g, h, j)).to eq([[6, 8, 10, 12, 14], [5, 12, 21, 32, 45], [0, 0, 0, 0, 0], [-4, -4, -4, -4, -4]])
        end
      end
    end

    %i[float32 float64].each do |dtype|
      context "#{dtype}" do
        specify do
          a = tf.constant([1, 2, 3, 4, 5], dtype: dtype)
          b = tf.constant([5, 6, 7, 8, 9], dtype: dtype)
          f = a + b
          g = a * b
          h = a / b
          j = a - b
          expect(f.dtype).to eq(dtype)
          expect(tr(sess.run(f, g, h, j))).to eq([[6.0, 8.0, 10.0, 12.0, 14.0], [5.0, 12.0, 21.0, 32.0, 45.0], [0.2, 0.3333, 0.4286, 0.5, 0.5556], [-4.0, -4.0, -4.0, -4.0, -4.0]])
        end
      end
    end
  end
end