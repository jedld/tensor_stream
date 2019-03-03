require "spec_helper"
require "benchmark"

RSpec.describe TensorStream::Session do
  let(:tf) { TensorStream }
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
      sess = TensorStream.session
      expect(sess.run(total)).to eq(7.0)
      expect(sess.run(product)).to eq(15.0)
    end

    it "different values on multiple runs" do
      tf.set_random_seed(1234)
      vec = tf.random_uniform([3])
      out1 = vec + 1
      out2 = vec + 2

      sess = tf.session(:ruby_evaluator)
      expect(sess.run(out1)).to eq([1.1915194503788924, 1.6221087710398319, 1.4377277390071144])
      expect(sess.run(out2)).to eq([2.7853585837137693, 2.7799758081188033, 2.2725926052826417])
      expect(sess.run(vec)).to eq([0.2764642551430967, 0.8018721775350193, 0.9581393536837052])
    end

    it "uniform values on a single run" do
      vec = TensorStream.random_uniform([3], seed: 1234)
      out1 = vec + 1
      out2 = vec + 2

      sess = TensorStream.session(:ruby_evaluator)
      expect(sess.run(out1, out2)).to eq([[1.1915194503788924, 1.6221087710398319, 1.4377277390071144], [2.1915194503788924, 2.622108771039832, 2.4377277390071144]])
    end

    it "can assign placeholders" do
      x = TensorStream.placeholder(TensorStream::Types.float32)
      y = TensorStream.placeholder(TensorStream::Types.float32)
      p = TensorStream.placeholder(TensorStream::Types.float32, name: "my_placeholder")
      z = x + y + p
      sess = TensorStream.session(:ruby_evaluator)

      expect(sess.run(z, feed_dict: {x => 3, y => 4.5, "my_placeholder" => 1})).to eql(8.5)
      expect(sess.run(z, feed_dict: {x => [1, 3], y => [2, 4], "my_placeholder" => [1, 1]})).to eql([4.0, 8.0])
    end

    context "#close" do
      it "closes session and releases resources" do
        sess = TensorStream.session
        sess.close
        expect(sess.closed?).to be
      end
    end

    context "#list_devices" do
      let(:sess) { TensorStream.session }
      it "list available device sin this session" do
        expect(sess.list_devices.map(&:name)).to include "cpu"
      end
    end

    xit "evaluate all while retaining some variables" do
      session = TensorStream::Session.default_session
      x = TensorStream.variable(1.0, :float32)
      y = TensorStream.variable(2.0, :float32)

      expression = TensorStream.sin(x) + TensorStream.cos(y)
      session.run(TensorStream.global_variables_initializer)
      partial_eval = session.run(expression, retain: [x])
      expect(partial_eval.to_math).to eq("(sin(Variable:0) + -0.4161468365471424)")
    end

    context "exceptions" do
      specify "checks for missing placeholder exceptions" do
        session = TensorStream::Session.default_session
        x = tf.placeholder(:float32)
        a = tf.constant(1.0) + x
        session.run(a, feed_dict: {x => 1})
        expect {
          session.run(a)
        }.to raise_error TensorStream::ValueError
      end
    end
  end
end
