require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Operation do

  let(:tf) { TensorStream } # allow calls to look like tensorflow
  let(:sess) { tf.session }

  # determine constant operations to aid evaluator in optimizing the graph
  specify "constant inference" do
    x = tf.placeholder(:float32)
    y = tf.variable(2.0, name: 'y')
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
end