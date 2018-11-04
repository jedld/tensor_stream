require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Tensor do

  let(:tf) { TensorStream }
  before(:each) do
    described_class.reset_counters
    TensorStream::Operation.reset_counters
    TensorStream::Graph.create_default
    TensorStream::Session.default_session.clear_session_cache
  end

  describe "Tensors" do
    it "can define Rank 0 Tensor definitions" do
      a = TensorStream.constant(3.0, dtype: TensorStream::Types.float32)
      b = TensorStream.constant(4.0)
      c = TensorStream.constant(4.0)
      d = TensorStream.variable(451, dtype: TensorStream::Types.int16)
      e = TensorStream.variable(451.12)
      total = a + b + c
      f = -e
      g = -d
      # expect(TensorStream::Graph.get_default_graph.nodes.keys).to eq([])
      expect(a.to_s).to eq("Const")
      expect(b.to_s).to eq("Const_1")
      expect(c.to_s).to eq("Const_2")
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
        c = TensorStream.constant([1, 2, 3, 4], dtype: :float32)
        expect(c.eval).to eql([1.0, 2.0, 3.0, 4.0])
      end
    end

    it "makes sure passed arrays are dense" do
      expect {
        TensorStream.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],[2.0]], name: 'a')
      }.to raise_exception TensorStream::ValueError

      expect {
        TensorStream.constant([[[1.0, 2.0],[3.0,4.0]], [[2.0],[4.0,5.0]]], name: 'b')
      }.to raise_exception TensorStream::ValueError
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

      expect(a.to_s).to eq("Const")
      expect(a.shape.to_s).to eq("TensorShape([Dimension(1)])")
      expect(b.shape.to_s).to eq("TensorShape([Dimension(0)])")
    end
  end

  describe "#rank" do
    it "correctly gives the rank" do
      a = TensorStream.constant(3.0, dtype: TensorStream::Types.float32)
      b = TensorStream.constant([3.0], dtype: TensorStream::Types.float32)
      c = TensorStream.constant([[3.0],[1.0]])
      d = TensorStream.constant([[[3.0,2.0]],[[1.0, 1.1]]], dtype: TensorStream::Types.float32)
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

  describe "#consumers" do
    it "lists dependent nodes to this tensor" do
      a = tf.constant([1,2,3,4,5])
      b = tf.constant([1,2,3,4,5])
      f = a * 2
      g = f + b
      expect(a.consumers.to_a).to eq(["mul:0", "add_1:0"])
      expect(b.consumers.to_a).to eq(["add_1:0"])
      expect(f.consumers.to_a).to eq(["add_1:0"])
      expect(g.consumers.to_a).to eq([])
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
      c = TensorStream.constant(3.0, dtype: TensorStream::Types.float64)
      d = TensorStream.constant("Hello", dtype: TensorStream::Types.string)
      e = TensorStream.constant(true, dtype: TensorStream::Types.boolean)
      expect(a.dtype).to eq(:int16)
      expect(b.dtype).to eq(:float32)
      expect(c.dtype).to eq(:float64)
      expect(d.dtype).to eq(:string)
      expect(e.dtype).to eq(:boolean)
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
      c_0 = tf.constant(0, name: "c")
      expect(c_0.name).to eq("c")

      c_1 = tf.constant(2, name: "c")
      expect(c_1.name).to eq("c_1")
    end
  end

  describe "operation shape inference" do
    it "operations can infer the possible shape of its outputs" do
      a = tf.constant([1.0, 2.0, 3.0, 4.0])
      b = tf.constant(1.0)
      c = a * b
      expect(c.shape.shape).to eq([4])

      m = tf.constant([[1.0, 0.5], [0.4, 0.2], [1.1, 1.2], [0.2, 0.1]])
      c = tf.constant([1.0, 1.0])
      e = m * c
      expect(e.shape.shape).to eq([4, 2])
      s = tf.reshape(e, [2, -1])
      expect(s.shape.shape).to eq([2, 4])
    end

    it "inferred size using reshape" do
      m = tf.constant([[1.0, 0.5], [0.4, 0.2], [1.1, 1.2], [0.2, 0.1]])
      s = tf.reshape(m, [2, -1])
      expect(s.shape.shape).to eq([2, 4])
    end

    # it "open shapes are also inferred" do
    #   a = tf.placeholder(:float32, shape: [nil, 4])
    #   m = tf.constant([[1.0, 0.5], [0.4, 0.2], [1.1, 1.2], [0.2, 0.1]])
    #   b = tf.constant(1.0)
    #   f = m.dot(a) + b
    #   expect(f.shape.shape).to eq([4, 4])
    # end

    it "handles reduction functions" do
      a = tf.constant([[1.0, 0.5], [0.4, 0.2], [1.1, 1.2], [0.2, 0.1]])
      f = tf.reduce_sum(a)
      expect(f.shape.shape).to eq([])
      f = tf.reduce_sum(a, 0)
      expect(f.shape.shape).to eq([2])
      f = tf.reduce_sum(a, 1)
      expect(f.shape.shape).to eq([4])
      expect(a[0].shape.shape).to eq([2])
    end
  end

  context ".floor" do
    specify do
      t = tf.constant(2.0)
      u = tf.constant(2.2)
      expect(t.floor.run).to eq(2)
      expect(u.floor.run).to eq(2)
    end
  end

  context ".ceil" do
    specify do
      t = tf.constant(3.1)
      expect(t.ceil.run).to eq(4)
    end
  end

  context "monkey patch" do
    specify do
      expect(2 + 1).to eql(3)
      expect(2 - 1).to eql(1)
      expect(2 / 1).to eql(2)
      expect(2 * 1).to eql(2)
      expect(2 % 1).to eql(0)
    end

    specify "add float" do
      a = 2.0.t
      f = 1.0 + a
      f1 = 1.0 / a
      f2 = 1.0 - a
      f3 = 1.0 * a
      f4 = 1.0 % a
      expect(f.run).to eql(3.0)
      expect(f1.run).to eql(2.0)
      expect(f2.run).to eql(-1.0)
      expect(f3.run).to eql(2.0)
      expect(f4.run).to eql(1.0)
    end

    specify "add int" do
      f = 1 + 2.t
      expect(f.run).to eql(3)
    end

    context "arrays" do
      specify do
        f = [1.0, 2.0, 3.0, 1.1, 2.2].t + 1.5
        expect(f.run).to eql([2.5, 3.5, 4.5, 2.6, 3.7])
      end
      specify do
        f = [1.0, 2.0, 3.0, 1.1, 2.2] + 1.0.t
        expect(f.run).to eql([2.0, 3.0, 4.0, 2.1, 3.2])
      end
    end
  end
end