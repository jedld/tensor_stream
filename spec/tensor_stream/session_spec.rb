require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Session do
  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    TensorStream::Graph.create_default
  end

  context "#run" do
    it "can execute operations" do
      a = TensorStream.constant(3.0, dtype: TensorStream::Types.float32)
      b = TensorStream.constant(4.0) # also tf.float32 implicitly
      c = TensorStream.constant(5.0)
      total = a + b
      product = a * c
      sess = TensorStream.Session
      expect(sess.run(total)).to eq(7.0)
      expect(sess.run(product)).to eq(15.0)
    end

    it "different values on multiple runs" do
      srand(1234)
      vec = TensorStream.random_uniform([3])
      out1 = vec + 1
      out2 = vec + 2

      sess = TensorStream.Session
      expect(sess.run(out1)).to eq([1.1915194503788924, 1.6221087710398319, 1.4377277390071144])
      expect(sess.run(out2)).to eq([2.7853585837137693, 2.7799758081188033, 2.2725926052826417])
      expect(sess.run(vec)).to eq([0.2764642551430967, 0.8018721775350193, 0.9581393536837052])
    end

    it "uniform values on a single run" do
      srand(1234)
      vec = TensorStream.random_uniform([3])
      out1 = vec + 1
      out2 = vec + 2

      sess = TensorStream.Session
      expect(sess.run(out1, out2)).to eq([[1.1915194503788924, 1.6221087710398319, 1.4377277390071144], [2.1915194503788924, 2.622108771039832, 2.4377277390071144]])
    end

    it "can assign placeholders" do
      x = TensorStream.placeholder(TensorStream::Types.float32)
      y = TensorStream.placeholder(TensorStream::Types.float32)
      z = x + y
      sess = TensorStream.Session
      expect(sess.run(z, feed_dict: { x =>  3, y => 4.5})).to eq(7.5)
      expect(sess.run(z, feed_dict: { x => [1, 3], y=> [2, 4]})).to eq([3, 7])
    end

    xit "evaluate all while retaining some variables" do
      session = TensorStream::Session.default_session
      x = TensorStream.Variable(1.0, :float32)
      y = TensorStream.Variable(2.0, :float32)

      expression = TensorStream.sin(x) + TensorStream.cos(y)
      session.run(TensorStream.global_variables_initializer)
      partial_eval = session.run(expression, retain: [x])
      expect(partial_eval.to_math).to eq("(sin(Variable:0) + -0.4161468365471424)")
    end
  end
end