require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Tensor do

  before(:each) do
    described_class.reset_counters
    TensorStream::Operation.reset_counters
    TensorStream::Graph.create_default
  end

  describe "Tensors" do
    it "can define Rank 0 Tensor definitions" do
      a = TensorStream.constant(3.0, dtype: TensorStream::Types.float32)
      b = TensorStream.constant(4.0)
      c = TensorStream.constant(4.0)
      d = TensorStream.Variable(451, dtype: TensorStream::Types.int16)
      e = TensorStream.Variable(451.12)
      total = a + b + c
      f = -e
      g = -d
      # expect(TensorStream::Graph.get_default_graph.nodes.keys).to eq([])
      expect(a.to_s).to eq("Const:0")
      expect(b.to_s).to eq("Const_1:0")
      expect(c.to_s).to eq("Const_2:0")
      expect(total.to_s).to eq("add_3:0")
      expect(d.to_s).to eq("Variable:0")
      expect(e.to_s).to eq("Variable_2:0")
      expect(a.shape.to_s).to eq("TensorShape([])")
      expect(f.to_s).to eq("negate_4:0")
      expect(f.dtype).to eq(:float32)
      expect(g.dtype).to eq(:int16)
    end

    context "constants" do
      it "does some type casting if dtype is specified" do
        c = TensorStream.constant(1, dtype: :float32)
        expect(c.eval).to eql(1.0)
        c = TensorStream.constant(1, dtype: :float32, shape: [2, 2])
        expect(c.eval).to eql([[1.0, 1.0], [1.0, 1.0]])
      end
    end

    it "automatically adjusts based on shape" do
      b = TensorStream.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [3, 2], name: 'b')
      expect(b.eval).to eq(
        [
         [1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]
        ]
      )

      c = TensorStream.constant(0, shape: [2,2])
      expect(c.eval).to eq(
        [[0.0, 0.0], [0.0, 0.0]]
      )

      c = TensorStream.constant(1, shape: [2,3])
      expect(c.eval).to eq(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
      )
    end

    it "can define Rank 1 Tensor definitions" do
      a = TensorStream.constant([3.0], dtype: TensorStream::Types.float32)
      b = TensorStream.constant([])

      expect(a.to_s).to eq("Const:1")
      expect(a.shape.to_s).to eq("TensorShape([Dimension(1)])")
      expect(b.shape.to_s).to eq("TensorShape([Dimension(0)])")
    end
  end

  describe "#rank" do
    it "correctly gives the rank" do
      a = TensorStream.constant(3.0, dtype: TensorStream::Types.float32)
      b = TensorStream.constant([3.0], dtype: TensorStream::Types.float32)
      c = TensorStream.constant([[3.0],[1.0]])
      d = TensorStream.constant([[[3.0,2.0]],[[1.0]]], dtype: TensorStream::Types.float32)
      expect(a.rank).to eq(0)
      expect(b.rank).to eq(1)
      expect(c.rank).to eq(2)
      expect(c.shape.to_s).to eq("TensorShape([Dimension(2),Dimension(1)])")
      expect(d.shape.to_s).to eq("TensorShape([Dimension(2),Dimension(1),Dimension(2)])")
    end
  end

  describe "#[]" do
    it "access indexes" do
      b = TensorStream.constant([3.0], dtype: TensorStream::Types.float32)
      expect(b[0].to_s).to eq("index:0")
    end
  end

  describe "#shape" do
    it "gives the shape of the tensor" do
      b = TensorStream.constant([3.0], dtype: TensorStream::Types.float32)
      expect(b.shape[0]).to eq(1)
    end

    it "create a vector of zeros with the same size as the number of columns in a given matrix" do
      my_matrix = TensorStream.constant([[1.0,1.0], [1.0, 1.0]])
      zeros = TensorStream.zeros(my_matrix.shape[1])
      expect(zeros.eval).to eq([0.0, 0.0])
    end
  end

  describe "#dtype" do
    it "returns the tensor's datatype" do
      a = TensorStream.constant(3.0, dtype: TensorStream::Types.int16)
      b = TensorStream.constant(3.0, dtype: TensorStream::Types.float32)
      expect(a.dtype).to eq(:int16)
      expect(b.dtype).to eq(:float32)
    end
  end

  describe "#eval" do
    it "evaluates a tensor" do
      constant = TensorStream.constant([1, 2, 3])
      tensor = constant * constant
      expect(tensor.eval()).to eq([1, 4, 9])
    end
  end

  describe "#eval" do
    it "evaluates tensor to its ruby equivalent value" do
      a = TensorStream.constant([3.0, 1.0], dtype: TensorStream::Types.float32)
      expect(a.eval).to eq([3.0, 1.0])
    end
  end

  describe "placeholders" do
    it "evalutes placeholders" do
      x = TensorStream.placeholder(TensorStream::Types.float32)
      y = TensorStream.placeholder(TensorStream::Types.float32)
      z = x + y
      expect(z.eval(feed_dict: { x =>  3.0, y => 4.5})).to eq(7.5)
    end

    specify "placeholders can have types" do
      x = TensorStream.placeholder(TensorStream::Types.float32)
      expect(x.dtype).to eq(:float32)
      expect(x.data_type).to eq(:float32)
    end
  end

  describe "naming operation" do
    it "can specify a name" do
      c_0 = TensorStream.constant(0, name: "c")
      expect(c_0.name).to eq("c")

      c_1 = TensorStream.constant(2, name: "c")
      expect(c_1.name).to eq("c_1")
    end
  end

  describe "tensor reshaping" do
    it "can reshape a tensor" do
    end
  end
end