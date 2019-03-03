require "spec_helper"
require "benchmark"

RSpec.describe TensorStream::Operation do
  let(:tf) { TensorStream } # allow calls to look like tensorflow
  let(:sess) { tf.session }

  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    tf.reset_default_graph
    sess.clear_session_cache
  end

  # determine constant operations to aid evaluator in optimizing the graph
  specify "constant inference" do
    x = tf.placeholder(:float32)
    y = tf.variable(2.0, name: "y")
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    c = a + b
    d = tf.sin(c)
    f = d + x
    g = a + y
    h = tf.cos(g)

    expect(d.is_const).to equal(true)
    expect(c.is_const).to equal(true)
    expect(f.is_const).to equal(false)
    expect(g.is_const).to equal(false)
    expect(h.is_const).to equal(false)
  end

  xcontext ".to_math" do
    specify "generates a math string for debugging" do
      a = tf.constant(1.0)
      b = tf.constant(2.0)
      c = tf.constant(1.0)

      f = tf.sin(tf.cos((a * b % tf.ones_like(c) - c)**2)) / 1
      expect(f.to_math(true).delete("\n").delete(" ")).to eq("(sin(cos(((mod(Const_1,ones_like(Const_2))-Const_2)^Const:0)))/Const_1:0)")
    end
  end
end
