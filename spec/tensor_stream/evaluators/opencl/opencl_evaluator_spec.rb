require "spec_helper"
require 'benchmark'
require 'tensor_stream/evaluator/opencl/opencl_evaluator'

RSpec.describe TensorStream::Evaluator::OpenclEvaluator do
  let(:tf) { TensorStream }
  let(:sess) { TensorStream.session([:opencl_evaluator, :ruby_evaluator]) }
  let(:instance) { described_class.new(sess, {})}

  it_behaves_like "standard ops evaluator"
  it_behaves_like "TensorStream::Train::Saver"

  def create_session
    TensorStream.session([:opencl_evaluator, :ruby_evaluator])
  end

  around :each do |spec|
    device = described_class.query_supported_devices.first
    # puts "using #{device.name}"
    tf.device(device) do
      spec.call
    end
  end

  context "supported ops" do
    specify do
      expect(described_class.ops.keys.size).to eq(58)
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
        check_numerics
        cond
        cos
        div
        equal
        exp
        flow_group
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
        not_equal
        pow
        print
        prod
        rank
        reciprocal
        reshape
        round
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
        transpose
        truncate
        where
      ])
    end

    it "allows automatic fallback" do
      a = tf.constant([1,2,3,4], dtype: :float32)
      c = tf.concat(a, 0)
      d = tf.sin(c)
      expect(sess.run(d)).to eq(-0.5440210700035095)
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