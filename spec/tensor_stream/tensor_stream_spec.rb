require "spec_helper"
require "benchmark"

RSpec.describe TensorStream do
  let(:tf) { TensorStream }

  before do
    TensorStream.disable_eager_execution
  end

  describe ".VERSION" do
    it "returns the version" do
      expect(TensorStream.version).to eq("1.0.3")
    end
  end

  xdescribe ".enable_eager_execution" do
    it "enables eager execution" do
      TensorStream.enable_eager_execution
      expect(TensorStream.executing_eagerly?).to be
      a = TensorStream.constant(2)
      b = TensorStream.constant(3)
      print("a = %i" % a)
      print("b = %i" % b)

      x = [[2.0]]
      m = TensorStream.matmul(x, x)
      expect(tr(m.to_a)).to eq([[4.0]])

      d = TensorStream.constant(3.1)
      expect(tr(d.to_f)).to eq(3.1)
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

  context ".variable_scope" do
    it "allows to define prefixes" do
      tf.variable_scope("foo") do
        tf.variable_scope("bar") do
          x = tf.zeros([], name: "qux")
          expect(x.name).to eq("foo/bar/qux")
        end
      end
    end

    specify "set default initializer" do
      tf.set_random_seed(1234)
      tf.variable_scope(initializer: tf.random_normal([2, 2])) do
        v1 = tf.get_variable("test", shape: [2, 2])
        expect(v1.name).to eq("test")
        sess = tf.session(:ruby_evaluator)
        sess.run(tf.global_variables_initializer)
        expect(sess.run(v1)).to eq([[0.5011628459350929, 1.301972948852967], [-1.621722019401658, 0.6690221526288901]])
      end
    end
  end
end
