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