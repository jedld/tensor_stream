require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream do
  before do
    TensorStream.disable_eager_execution
  end

  describe ".VERSION" do
    it "returns the version" do
      expect(TensorStream.version).to eq("0.1.0")
    end
  end

  describe ".enable_eager_execution" do
    it "enables eager execution" do
      TensorStream.enable_eager_execution
      expect(TensorStream.executing_eagerly?).to be
      a = TensorStream.constant(2)
      b = TensorStream.constant(3)
      print("a = %i" % a)
      print("b = %i" % b)

      x = [[2.0]]
      m = TensorStream.matmul(x, x)
      expect(m.to_a).to eq([[4.0]])

      d = TensorStream.constant(3.1)
      expect(d.to_f).to eq(3.1)
    end
  end

  describe ".random_uniform" do
    context "shape (3,)" do
      it "Creates an operation to generate a random set of values of the given shape" do
        srand(1234567)
        vec = TensorStream.random_uniform([3])
        expect(vec.eval).to eq([0.23702916849534994, 0.007648373861731117, 0.019830308342374425])

        #evaluating again generates new values
        expect(vec.eval).to eq([0.3130926186495132, 0.09945466414888471, 0.1951742921107925])
      end
    end

    context "shape (2, 2)" do
      it "Creates an operation to generate a random set of values of the given shape" do
        srand(1234567)
        vec = TensorStream.random_uniform([2,2])
        expect(vec.eval).to eq([[0.23702916849534994, 0.007648373861731117], [0.019830308342374425, 0.3130926186495132]])

        #evaluating again generates new values
        expect(vec.eval).to eq([[0.09945466414888471, 0.1951742921107925], [0.20729802198472924, 0.16493119121125488]])
      end
    end
  end
end