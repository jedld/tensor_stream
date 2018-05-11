require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Operation do
  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    TensorStream::Graph.create_default
    srand(1234)
  end

  let(:tf) { TensorStream } # allow calls to look like tensorflow
  let(:sess) { tf.Session }

  context ".zeros_like" do
    it "Creates a tensor with all elements set to zero." do
      tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
      z = tf.zeros_like(tensor)
      expect(z.eval).to eq([[0, 0, 0], [0, 0, 0]])
    end
  end

  context ".concat" do
    it "Concatenates tensors along one dimension." do
      t1 = [[1, 2, 3], [4, 5, 6]]
      t2 = [[7, 8, 9], [10, 11, 12]]
      expect(tf.concat([t1, t2], 0).eval).to eq([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
      expect(tf.concat([t1, t2], 1).eval).to eq([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]])
    end

    it "negative axis" do
      t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
      t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
      expect(tf.concat([t1, t2], -1).eval).to eq(
      [[[ 1,  2,  7,  4],
        [ 2,  3,  8,  4]],
       [[ 4,  4,  2, 10],
        [ 5,  3, 15, 11]]])
    end
  end

  context ".reshape" do
    it "Reshapes a tensor." do
      t = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      expect(tf.reshape(t, [3, 3]).eval).to eq(
        [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])

      t = [[[1, 1], [2, 2]],
           [[3, 3], [4, 4]]]

      expect(tf.reshape(t, [2, 4]).eval).to eq([[1, 1, 2, 2],
        [3, 3, 4, 4]])
    end

    it "reshape to scalar" do
      t = [7]
      expect(tf.reshape(t, []).eval).to eq(7)
    end

    it "flattens a tensor" do
      t = [[[1, 1, 1],
            [2, 2, 2]],
          [[3, 3, 3],
          [4, 4, 4]],
          [[5, 5, 5],
          [6, 6, 6]]]
      expect(tf.shape(t).eval).to eq([3, 2, 3])
      expect(tf.reshape(t, [-1]).eval).to eq([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])
      expect(tf.reshape(t, [2, -1]).eval).to eq([[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]])
    end

    it "should fail if dimensions do not match" do
      t = [[[1, 1, 1],
            [2, 2, 2]],
          [[3, 3, 3],
          [4, 4, 4]],
          [[5, 5, 5],
          [6, 6, 6]]]
      expect {
        tf.reshape(t,[3,2,2]).eval
      }.to raise_exception

    end

    it "inference" do
      t = [[[1, 1, 1],
            [2, 2, 2]],
            [[3, 3, 3],
            [4, 4, 4]],
            [[5, 5, 5],
            [6, 6, 6]]]

      expect(tf.reshape(t, [-1, 9]).eval).to eq([[1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]])
      
      expect(tf.reshape(t, [ 2, -1, 3]).eval).to eq(
        [[[1, 1, 1],
          [2, 2, 2],
          [3, 3, 3]],
          [[4, 4, 4],
          [5, 5, 5],
          [6, 6, 6]]])
    end
  end

  context ".max" do
    it "returns the maximum of two tensors" do
      a = tf.constant(1.0)
      b = tf.constant([1.0, 3.0])
      d = tf.constant([3.0, 1.1])
      c = tf.constant(2.1)
      expect(tf.max(a,c).eval).to eq(2.1)
      expect(tf.max(b,d).eval).to eq([3.0, 3.0])
    end

    xit "computes for the gradient" do
      b = tf.constant([1.0, 3.0])
      d = tf.constant([3.0, 1.1])
      g = tf.gradients(tf.max(b,d), [b, d])
      expect(g.eval).to eq([])
    end
  end

  context ".cast" do
    it "converts from one datatype to another" do
      a = tf.constant([1.0, 3.0])
      b = tf.constant([true, true])
      expect(tf.cast(a, :int32).eval).to eql([1, 3])
      expect(tf.cast(a, :boolean).eval).to eql([true, true])
      expect(tf.cast(b, :float32).eval).to eql([1.0, 1.0])
      expect(tf.cast(b, :int32).eval).to eql([1, 1])
    end
  end

  context ".equal" do
    it "returns the truth value of two tensors" do
      a = tf.constant(1.0)
      b = tf.constant(1.0)
      c = tf.constant(2.1)
      d = tf.constant([[1.0]])
      e = tf.constant([[1.0]])
      f = tf.constant([[2.0]])
      expect(tf.equal(a, b).eval).to eq(true)
      expect(tf.equal(a, c).eval).to eq(false)
      expect(tf.equal(d, e).eval).to eq([[true]])
      expect(tf.equal(e, f).eval).to eq([[false]])

      expect((a == b).eval).to eq(true)
      expect((a == c).eval).to eq(false)
    end
  end

  context ".not_equal" do
    it "returns the truth value of two tensors" do
      a = tf.constant(1.0)
      b = tf.constant(1.0)
      c = tf.constant(2.1)
      d = tf.constant([[1.0]])
      e = tf.constant([[1.0]])
      f = tf.constant([[2.0]])
      expect(tf.not_equal(a, b).eval).to eq(false)
      expect(tf.not_equal(a, c).eval).to eq(true)
      expect(tf.not_equal(d, e).eval).to eq([[false]])
      expect(tf.not_equal(e, f).eval).to eq([[true]])

      expect((a != b).eval).to eq(false)
      expect((a != c).eval).to eq(true)
    end
  end

  # Outputs random values from a uniform distribution.
  # The generated values follow a uniform distribution in the range [minval, maxval). The lower bound minval is included in the range, while the upper bound maxval is excluded.
  # For floats, the default range is [0, 1). For ints, at least maxval must be specified explicitly.
  # In the integer case, the random integers are slightly biased unless maxval - minval is an exact power of two. The bias is small for values of maxval - minval significantly smaller than the range of the output (either 2**32 or 2**64).
  context ".random_uniform" do
    [
      [[],     0.1915194503788923,       0.3830389007577846         ],
      [[1],   [0.1915194503788923],      [0.3830389007577846]         ],
      [[2,3], [[0.1915194503788923, 0.6221087710398319, 0.4377277390071145], [0.7853585837137692, 0.7799758081188035, 0.2725926052826416]],  [[0.3830389007577846, 1.2442175420796637, 0.875455478014229], [1.5707171674275384, 1.559951616237607, 0.5451852105652832]] ]
    ].each do |shape, expected, range_expected|
      describe "shape #{shape}" do
        it "generates random uniform values" do
          expect(TensorStream.random_uniform(shape).eval).to eq(expected)
        end

        specify "with ranges" do
          expect(TensorStream.random_uniform(shape, minval: 0, maxval: 2).eval).to eq(range_expected)
        end
      end
    end
  end

  context ".random_normal" do
    [
      [[], 0.5011628459350929],
      [[1],   [0.5011628459350929] ],
      [[2,3], [[0.5011628459350929, 1.301972948852967, -1.621722019401658], [0.6690221526288901, 0.14937983113945622, -0.783723693080629]] ],
    ].each do |shape, expected|
      describe "shape #{shape}" do
        it "generates random normal values" do
          expect(TensorStream.random_normal(shape).eval).to eq(expected)
        end
      end
    end
  end

  context '.argmax' do
    it 'Returns the index with the largest value across axes of a tensor. (deprecated arguments)' do
      a = tf.constant([
        [31, 23,  4, 24, 27, 34],
        [18,  3, 25,  0,  6, 35],
        [28, 14, 33, 22, 20,  8],
        [13, 30, 21, 19,  7,  9],
        [16,  1, 26, 32,  2, 29],
        [17, 12,  5, 11, 10, 15]])

      b = tf.constant([1,2,3,4,5,6])
      expect(tf.argmax(a).eval).to eq([0, 3, 2, 4, 0, 1])
      expect(tf.argmax(a, 1).eval).to eq([5, 5, 2, 1, 3, 0])
      expect(tf.argmax(a, 0).eval).to eq([0, 3, 2, 4, 0, 1])
      expect(tf.argmax(b, 0).eval).to eq(5)
      expect(tf.argmax(b, 0, output_type: :float32).eval).to eql(5.0)
    end
  end

  context ".zeros" do
    it "generates a zero tensor" do
      a = tf.zeros([2,2])
      expect(a.eval).to eq([[0.0, 0.0], [0.0, 0.0]])
    end
  end
    
  context ".zeros_like" do
    it "generates a zero tensor based on another tensor" do
      a = tf.zeros_like([2,2,2,2,2])
      b = tf.zeros_like([[2,2],[3,3]])
      expect(a.eval).to eq([0, 0, 0, 0, 0])
      expect(b.eval).to eq([[0, 0], [0, 0]])
    end
  end

  context ".ones_like" do
    it "generates a zero tensor based on another tensor" do
      a = tf.ones_like([2, 2, 2, 2, 2])
      b = tf.ones_like([[2, 2],[3, 3]])
      expect(a.eval).to eq([1, 1, 1, 1, 1])
      expect(b.eval).to eq([[1, 1], [1, 1]])
    end
  end

  context ".ones" do
    it "generates a ones tensor" do
      ones = tf.ones([2,2])
      expect(ones.eval).to eq([[1.0, 1.0], [1.0, 1.0]])
    end
  end

  context ".reduce_sum" do
    it "computes the sum of elements across dimensions of a tensor." do
      x = tf.constant([[1, 1, 1], [1, 1, 1]])
      expect(tf.reduce_sum(x).eval).to eq(6)
      expect(tf.reduce_sum(x, 0).eval).to eq([2, 2, 2])
      expect(tf.reduce_sum(x, 1).eval).to eq([3, 3])
      expect(tf.reduce_sum(x, 1, keepdims: true).eval).to eq([[3], [3]])
      expect(tf.reduce_sum(x, [0, 1]).eval).to eq(6)
    end

    specify "computes the gradients properly" do
      a = tf.constant([[1,2,3],[4,5,6]])
      op = tf.reduce_sum(a)
      expect(tf.gradients(op,[a]).eval).to eq([[[1, 1, 1], [1, 1, 1]]])
    end
  end

  context ".reduce_prod" do
    it "computes the sum of elements across dimensions of a tensor." do
      x = tf.constant([[2, 1, 2], [2, 1, 2]])
      expect(tf.reduce_prod(x).eval).to eq(16)
      expect(tf.reduce_prod(x, 0).eval).to eq([4, 1, 4])
      expect(tf.reduce_prod(x, 1).eval).to eq([4, 4])
      expect(tf.reduce_prod(x, 1, keepdims: true).eval).to eq([[4], [4]])
      expect(tf.reduce_prod(x, [0, 1]).eval).to eq(16)
    end

    it "reduceing an empty array" do
      x = tf.constant([])
      y = tf.constant([[], []])
      expect(tf.reduce_prod(x).eval).to eq(1.0)
      expect(tf.reduce_prod(y, 0).eval).to eq([])
      expect(tf.reduce_prod(y, 1).eval).to eq([1.0, 1.0])
    end

    xspecify "computes the gradients properly" do
      a = tf.constant([[1,2,3],[4,5,6]])
      op = tf.reduce_prod(a)
      expect(tf.gradients(op,[a]).eval).to eq([[720, 360, 240],[180, 144, 120]])
    end
  end

  context ".pow" do
    it "Computes the power of tensor x to tensor y" do
      x = tf.constant([[2, 2], [3, 3]])
      y = tf.constant([[8, 16], [2, 3]])
      p = tf.pow(x, y)  # [[256, 65536], [9, 27]]
      expect(sess.run(p)).to eq([[256, 65536], [9, 27]])

      p = tf.pow(x, 2)
      expect(sess.run(p)).to eq([[4, 4], [9, 9]])
    end

    it "gradients of the power rule" do
      x = tf.constant([[1.1, 1.3], [1.3, 1.2]])
      y = tf.constant([[1.5, 2.0], [1.1, 2.0]])
      p = tf.pow(x, y)  # [[256, 65536], [9, 27]]
      g = tf.gradients(p, [x, y])
      expect(tr(sess.run(g))).to eq([
        [[1.5732, 2.6], [1.1292, 2.4]],
        [[0.11, 0.4434], [0.3501, 0.2625]]
      ])
    end
  end

  context ".pad" do
    it "pads a tensor, rank 1" do
      t = tf.constant([1, 2, 3])
      paddings = tf.constant([[1,1]])
      expect(tf.pad(t, paddings).eval).to eq([0, 1, 2, 3, 0])
    end

    it "pads a tensor, rank 2" do
      t = tf.constant([[1, 2, 3], [4, 5, 6]])
      paddings = tf.constant([[1, 1], [2, 2]])

      expect(tf.pad(t, paddings, mode: "CONSTANT").eval).to eq(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 2, 3, 0, 0],
         [0, 0, 4, 5, 6, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]]
      )

      paddings_2 = tf.constant([[0, 1], [0, 2]])
      expect(tf.pad(t, paddings_2, mode: "CONSTANT").eval).to eq(
        [
         [1, 2, 3, 0, 0],
         [4, 5, 6, 0, 0],
         [0, 0, 0, 0, 0]
        ]
      )

      paddings_3 = tf.constant([[1, 0], [2, 0]])
      expect(tf.pad(t, paddings_3, mode: "CONSTANT").eval).to eq(
        [[0, 0, 0, 0, 0],
         [0, 0, 1, 2, 3],
         [0, 0, 4, 5, 6]]
      )
    end
  end

  context ".print" do
    it "behaves like identity but prints a message to stdout" do
      x = tf.constant([[2.0, 2.0], [3.0, 3.0]])
      y = tf.print(x, x, message: "this is a prefix")
      z = tf.sin(y)
      expect(tr(z.eval)).to eq([[0.9093, 0.9093], [0.1411, 0.1411]])
    end
  end

  context ".slice" do
    it "slices a tensor" do
      t = tf.constant([[[1, 1, 1], [2, 2, 2]],
        [[3, 3, 3], [4, 4, 4]],
        [[5, 5, 5], [6, 6, 6]]])
      expect(tf.slice(t, [1, 0, 0], [1, 1, 3]).eval).to eq([[[3, 3, 3]]])
      expect(tf.slice(t, [1, 0, 0], [1, 2, 3]).eval).to eq([[[3, 3, 3], [4, 4, 4]]])
      expect(tf.slice(t, [1, 0, 0], [2, 1, 3]).eval).to eq([[[3, 3, 3]], [[5, 5, 5]]])
    end

    it "1D tensor slicing" do
      t  = tf.constant([1,2,3,4,5,6,7])
      expect(tf.slice(t, [2], [1]).eval).to eq([3])
    end
  end

  context ".rank" do
    it "returns the rank of a tensor" do
      t1 = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
      t2 = tf.constant(1)
      t3 = tf.constant([1,2])
      rank1 = tf.rank(t1)
      rank2 = tf.rank(t2)
      rank3 = tf.rank(t3)
      expect(rank1.eval).to eq(3)
      expect(rank2.eval).to eq(0)
      expect(rank3.eval).to eq(1)
    end
  end

  context ".negate" do
    it "computes the negative of a tensor" do
      x = tf.constant(0.1)
      y = tf.constant([[1.1, 16.1], [2.1, 3.0]])
      z = -tf.constant(4.1)
      x_negate = tf.negate(x)
      y_negate = tf.negate(y)

      expect(sess.run(x_negate)).to eq(-0.1)
      expect(sess.run(y_negate)).to eq([[-1.1, -16.1], [-2.1, -3.0]])
      expect(sess.run(z)).to eq(-4.1)
    end
  end

  # tests for single parameter algebra functions
[
  [:sin, 0.0998,   [[0.8912, -0.3821], [0.8632, 0.1411]],  0.995, [[0.4536, -0.9241], [-0.5048, -0.99]]                      ],
  [:cos, 0.995,    [[0.4536, -0.9241], [-0.5048, -0.99]], -0.0998, [[-0.8912, 0.3821], [-0.8632, -0.1411]]                   ],
  [:tan, 0.1003,   [[1.9648, 0.4134], [-1.7098, -0.1425]], 1.0101,  [[4.8603, 1.1709], [3.9236, 1.0203]]                     ],
  [:tanh, 0.0997,  [[0.8005, 1.0], [0.9705, 0.9951]],      0.9901, [[0.3592, 0.0], [0.0582, 0.0099]]                         ],
  [:log, -2.3026,  [[0.0953, 2.7788], [0.7419, 1.0986]],   10.0, [[0.9091, 0.0621], [0.4762, 0.3333]]                        ],
  [:exp, 1.1052,   [[3.0042, 9820670.9221], [8.1662, 20.0855]], 1.1052, [[3.0042, 9820670.9221], [8.1662, 20.0855]]          ],
  [:square, 0.01,  [[1.21, 259.21], [4.41, 9.0]],          0.2, [[2.2, 32.2], [4.2, 6.0]]                                    ],
  [:negate, -0.1,  [[-1.1, -16.1], [-2.1, -3.0]],         -1.0, [[-1.0, -1.0], [-1.0, -1.0]]                                 ],
  [:identity, 0.1, [[1.1, 16.1], [2.1, 3.0]],             1.0, [[1, 1], [1, 1]]                                              ],
  [:abs, 0.1,      [[1.1, 16.1], [2.1, 3.0]],             1.0, [[1, 1], [1, 1]]                                              ],
  [:sqrt, 0.3162,  [[1.0488, 4.0125], [1.4491, 1.7321]],   1.5811, [[0.4767, 0.1246], [0.345, 0.2887]]                       ],
].each do |func, scalar, matrix, gradient, gradient2|
  context ".#{func}" do
    let(:x) { tf.constant(0.1) }
    let(:y) {  tf.constant([[1.1, 16.1], [2.1, 3.0]]) }
    let(:f_x) { tf.send(func,x) }
    let(:f_y) { tf.send(func,y) }

    specify "scalar #{func} value" do
      expect(tr(sess.run(f_x))).to eq(scalar)
    end

    specify "matrix #{func} values" do
      expect(tr(sess.run(f_y))).to eq(matrix)
    end

    specify "gradient #{func} values" do
      grad = tf.gradients(f_x, [x]).first
      grad_2 = tf.gradients(f_y, [y]).first
      expect(tr(sess.run(grad))).to eq(gradient)
      expect(tr(sess.run(grad_2))).to eq(gradient2)
    end
  end
end

  context ".abs" do
    it "Computes the absolute value of a tensor" do
      tf = TensorStream

      a = [[1,2],[-1, 2], [3,-3]]
      b = -1.123

      expect(tf.abs(a).eval).to eq([[1, 2], [1, 2], [3, 3]])
      expect(tf.abs(b).eval).to eq(1.123)
    end

    specify "should compute for the gradient" do
      a = tf.constant([[1,2],[-1, 2], [3,-3]])
      expect(tf.gradients(tf.abs(a),[a]).eval).to eq([[[ 1,  1],
        [-1,  1],
        [ 1, -1]]])
    end
  end

  context ".sign" do
    it "Returns an element-wise indication of the sign of a number." do
      tf = TensorStream

      a = tf.constant([[1,2],[-1, 2], [3,-3]])
      b = -1.123

      expect(tf.sign(a).eval).to eq([[1, 1], [-1, 1], [1, -1]])
      expect(tf.sign(b).eval).to eq(-1.0)
    end
  end

  context ".matmul" do
    it "performs matrix multiplication" do
      a = tf.constant([1, 2, 3, 4, 5, 6], shape: [2, 3])
      b = tf.constant([7, 8, 9, 10, 11, 12], shape: [3, 2])
      c = tf.matmul(a, b)
      expect(c.eval).to eq([[ 58,  64],
                            [139, 154]])
      d = tf.matmul(a, b, transpose_a: true, transpose_b: true)
      expect(d.eval).to eq([[39, 49, 59], [54, 68, 82], [69, 87, 105]])
    end

    specify "gradients" do
      a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      b = tf.constant([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [10.0, 11.0, 12.0]])
      
      y = tf.matmul(a, tf.sin(b))

      expect(tr(y.eval)).to eq([[-2.0631, -4.0106, -2.2707], [-3.3563, -7.0425, -4.2538]])

      g = tf.gradients(y, [a, b])

      expect(tr(g.eval)).to eq([[[2.0585, -2.0806, -2.0806], [2.0585, -2.0806, -2.0806]], [[3.7695, -0.7275, -4.5557], [-5.8735, 0.031, 5.907], [-7.5516, 0.0398, 7.5947]]])
    end
  end

  context ".transpose" do
    it "transposes matrices" do
      tf.program do |tf|
        x = tf.constant([[1, 2, 3], [4, 5, 6]])
        t = tf.transpose(x)

        expect(sess.run(t)).to eq([[1, 4], [2, 5], [3, 6]])
      end
    end
  end

  context ".derivative" do
    it "Creates a derivative graph for a computation" do
      x = tf.placeholder(TensorStream::Types.float32)
      p = tf.pow(x, 3) 

      derivative_function = TensorStream::MathGradients.derivative(p, x)
      expect(p.eval(feed_dict: { x => 2})).to eq(8)
      expect(derivative_function.eval(feed_dict: { x => 2})).to eq(12)
  
      # f(x) = (sin x) ^ 3
      # dx = 3(sin x)^2 * cos x
      y = tf.sin(x) ** 3
      derivative_function_y = TensorStream::MathGradients.derivative(y, x)
      expect(derivative_function_y.eval(feed_dict: { x => 1 })).to eq(1.147721101851439)
    end
  end

  context ".shape" do
    it "returns a 1D tensor representing shape of target tensor" do
      tf.program do |tf|
        t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
        shape = tf.shape(t)
        expect(shape.eval).to eq([2, 2, 3])

        u = tf.constant(1)
        shape = tf.shape(u)
        expect(shape.eval).to eq([])

        v = tf.constant([[1,2,3],[4,5,6]])
        shape = tf.shape(v)
        expect(shape.eval).to eq([2 ,3])
      end
    end
  end

  context ".eye" do
    it "creates an identity matrix" do
      tf.program do |tf|
        e = tf.eye(2)
        expect(e.eval).to eq([[1.0, 0.0],[0.0, 1.0]])

        e = tf.eye(3)
        expect(e.eval).to eq([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        e = tf.eye(3, num_columns: 2)
        expect(e.eval).to eq([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      end
    end

    specify "using in matrix multiplication" do
      a = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
      b = tf.constant([[0.1, 0.1], [0.1, 0.1], [0.2, 0.2]])
      m = tf.matmul(a, b)
      expect(m.eval).to eq([[0.9000000000000001, 0.9000000000000001], [0.9000000000000001, 0.9000000000000001]])

      g = tf.gradients(m, [a])
      expect(g.eval).to eq([[[0.2, 0.2, 0.4], [0.2, 0.2, 0.4]]])

      d_wra = tf.matmul(tf.eye(a.shape[0]), b, transpose_b: true)
      expect(d_wra.eval).to eq([[0.1, 0.1, 0.2], [0.1, 0.1, 0.2]])
    end
  end

  context ".gradients" do
    it "Constructs symbolic derivatives of sum of ys w.r.t. x in xs." do
      a = tf.constant(0.0)
      b = a * 2
      g = tf.gradients(a + b, [a, b], stop_gradients: [a, b])
      h = tf.gradients(a + b, [a, b])

      expect(g.eval).to eq([1.0, 1.0])
      expect(h.eval).to eq([3.0, 1.0])
    end

    it "using stop gradients" do
      a = tf.stop_gradient(tf.constant(0.0))
      b = tf.stop_gradient(a * 2)
      h = tf.gradients(a + b, [a, b])
      expect((a+b).eval).to eq(0)
      expect((a+b).to_math).to eq("(0.0 + (0.0 * 2.0))")
      expect(h.eval).to eq([1.0, 1.0])
    end

    it "computes gradient of sin" do
      var = tf.constant(1.0) # Must be a tf.float32 or tf.float64 variable.
      loss = tf.sin(var) # some_function_of() returns a `Tensor`.
      var_grad = tf.gradients(loss, [var])[0]

      expect(var_grad.eval).to eq(0.5403023058681398)
    end
  end

  context ".cond" do
    it "returns a specific tensor function depending on the value of the predicate"  do
      x = tf.constant(2.0)
      y = tf.constant(3.0)
      z = tf.multiply(x, y)

      result = tf.cond(x < y, tf.add(x, z), tf.square(y))
      result2 = tf.cond(x > y, -> { tf.add(x, z) }, -> { tf.square(y) })
      expect(result.eval).to eq(8.0)
      expect(result2.eval).to eq(9.0)
    end

    it "supports gradients" do
      x = tf.constant(2.0)
      y = tf.constant(3.0)
      z = tf.multiply(x, y)

      result = tf.cond(x < y, tf.add(x, z), tf.square(y))
      result2 = tf.cond(x > y, tf.add(x, z), tf.square(y))

      grad1 = tf.gradients(result, [x, y])
      grad2 = tf.gradients(result2, [x, y])

      expect(grad1.eval).to eq([4.0, 2.0])
      expect(grad2.eval).to eq([0.0, 6.0])
    end
  end

  context ".where" do
    it "does an elementwise comparison and picks the appropriate element from x or y" do
      a = tf.constant([1,2,3,4,5])
      b = tf.constant([6,6,6,6,6])
      c = tf.constant([8,8,8,8,8])
      
      expect(tf.where(a > 3, b, c).eval).to eq([8, 8, 8, 6, 6])
    end

    it "supports gradients" do
      a = tf.constant([1,2,3,4,5])
      b = tf.constant([6,6,6,6,6])
      c = tf.constant([8,8,8,8,8])

      expr = tf.where(a > 3, b, c)
      g = tf.gradients(expr, [b, c])
      expect(g.eval).to eq([[0, 0, 0, 1, 1], [1, 1, 1, 0, 0]])
    end
  end

  context ".mul" do
    it "performs elementwise multiplication" do
      a = tf.constant([[1, 2, 3], [4, 5, 6]])
      c = a * 6
      expect(c.eval).to eq([[6, 12, 18], [24, 30, 36]])

      b = tf.constant([1, 2, 3])
      d = a * b
      expect(d.eval).to eq([[1, 4, 9], [4, 10, 18]])
    end

    it "constant multiplication" do
      a= tf.constant([[1, 2, 3], [4, 5, 6]])
      c = tf.constant(6) * a
      expect(a.eval).to eq([[1, 2, 3], [4, 5, 6]])

      b= tf.constant([1,2,3,4,5,6])
      d= tf.constant(6) * b
      expect(d.eval).to eq([6, 12, 18, 24, 30, 36])
    end

    it "handles two rank 1 tensors" do
      a = tf.constant([7.0, 7.0, 7.0, 7.0, 7.0])
      b = tf.constant([-0.1079, 2.281999999999999, 1.1489, -0.5005000000000001, -3.5218999999999996])
      c = a * b
      expect(c.eval).to eq([-0.7553, 15.973999999999993, 8.042300000000001, -3.5035000000000003, -24.653299999999998])
    end

    it "handles different rows" do
      a = tf.constant([[1.0, 1.0], [1.0, 1.0]])
      b = tf.constant([[4.0, 4.0]])
      c = a * b
      expect(c.eval).to eq([[4.0, 4.0], [4.0, 4.0]])
    end

    it "different rank multiplication" do
      a = tf.constant([7.0, 7.0, 7.0, 7.0, 7.0])
      b = tf.constant([
        [2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1]])
      c = a * b
      expect(c.eval).to eq([[14.0, 14.0, 14.0, 14.0, 14.0], [7.0, 7.0, 7.0, 7.0, 7.0]])
    end
  end

  context ".reduce_mean" do
    it "Computes the mean of elements across dimensions of a tensor" do
      x = tf.constant([[1.0, 1.0], [2.0, 2.0]])
      expect(tf.reduce_mean(x).eval).to eq(1.5)
      expect(tf.reduce_mean(x, 0).eval).to eq([1.5, 1.5])
      expect(tf.reduce_mean(x, 1).eval).to eq([1.0, 2.0])
    end

    it ".computes for the gradient" do
      x = tf.constant([[1.0, 1.0], [2.0, 2.0]])
      f = tf.reduce_mean(x)
      g = tf.gradients(f, [x])
      expect(g.eval).to eq([[[0.25, 0.25], [0.25, 0.25]]])
    end
  end

  context ".less" do
    it "returns true if a < b" do
      a = tf.constant(2.0)
      b = tf.constant(3.0)
      expect(tf.less(a, b).eval).to eq(true)
      expect(tf.less(b, a).eval).to eq(false)
    end
  end

  context ".greater" do
    it "returns true if a > b" do
      a = tf.constant(2.0)
      b = tf.constant(3.0)
      expect(tf.greater(a, b).eval).to eq(false)
      expect(tf.greater(b, a).eval).to eq(true)
    end

    it "handles rank 1 or higher" do
      a = tf.constant([[1.1, 1.3], [1.3, 1.2]])
      c = a > 0
      expect(c.eval).to eq([[true, true], [true, true]])
    end
  end

  context ".sub" do
    let(:a) { tf.constant([1.0, 2.0, 3.0])}
    let(:b) { tf.constant([0.1, 0.2, 0.3])}
    let(:c) { tf.constant(0.1) }
    let(:m) { tf.constant([[1.0, 2.0, 3.0], [2.0, 3.0 ,4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]) }

    it "substracts two arrays" do
      expect((a - b).eval).to eq([0.9, 1.8, 2.7])
    end

    it "substracts an array and a constant" do
      expect((a - c).eval).to eq([0.9, 1.9, 2.9])
    end

    it "substracts a matrix and an array" do
      expect((m - a).eval).to eq([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [4.0, 4.0, 4.0], [7.0, 7.0, 7.0]])
    end
  end

  context "multivariate functions" do
    let(:a)   { tf.constant(1.0) }
    let(:b)   { tf.constant(2.0) }
    let(:a_1) { tf.constant([1.0, 1.5]) }
    let(:b_1) { tf.constant([2.0, 0.1]) }
    let(:a_2) { tf.constant([[1.0, 1.5],[0.8,  0.2]]) }
    let(:b_2) { tf.constant([[2.0, 0.1],[3.0, 0.01]]) }

    def func_test(op, x, y, e1, e2)
      func = tf.send(op.to_sym, x, y)
      expect(tr(func.eval)).to eq(e1)
      grad = tf.gradients(func, [x, y])
      expect(tr(grad.eval)).to eq(e2)
    end

    [ 
      #op   rank 0   rank 1   rank 2   grad 0   grad 1  grad 2
      [:add, 3.0,  [3.0, 1.6],  [[3.0, 1.6], [3.8, 0.21]],    [1.0,  1.0],  [[1.0, 1.0], [1.0,   1.0]],  [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]] ],
      [:sub, -1.0, [-1.0, 1.4], [[-1.0, 1.4], [-2.2, 0.19]],  [1.0, -1.0],  [[1.0, 1.0], [-1.0, -1.0]],   [[[1.0, 1.0], [1.0, 1.0]], [[-1.0, -1.0], [-1.0, -1.0]]] ],
    ].each do |op, expected_0, expected_1, expected_2, expected_grad_0, expected_grad_1, expected_grad_2|
      context ".#{op}" do


        specify "basic scalar operation" do
          func_test(op, a, b, expected_0, expected_grad_0)
        end

        specify "basic rank 1 operation" do
          func_test(op, a_1, b_1, expected_1, expected_grad_1)
        end

        specify "basic rank 2 operation" do
          func_test(op, a_2, b_2, expected_2, expected_grad_2)
        end
      end
    end

    [
      [:add, [3.0, 3.5],   [[3.0, 3.5], [2.8, 2.2]], [[1.0, 1.0], 2.0],       [[[1.0, 1.0], [1.0, 1.0]], 4.0] ],
      [:sub, [-1.0, -0.5], [[-1.0, -0.5], [-1.2, -1.8]], [[1.0, 1.0], -2.0],  [[[1.0, 1.0], [1.0, 1.0]], -4.0] ],
    ].each do |op, expected_1_0, expected_2_0, grad_1_0, grad_2_0|
      context ".#{op}" do
        specify "mixed rank operation 1  vs 0" do
          func_test(op, a_1, b, expected_1_0, grad_1_0)
        end

        specify "mixed rank operation 2  vs 0" do
          func_test(op, a_2, b, expected_2_0, grad_2_0)
        end
      end
    end
  end

  context ".div" do
    let(:a) { tf.constant(2.5) }
    let(:b) { tf.constant(3.1) }

    it "divides to tensors" do
      op = a / b
      expect(tr(op.eval)).to eq(0.8065)
    end

    it "supports gradients" do
      grad = tf.gradients(a/b, [a,b])
      expect(tr(grad.eval)).to eq([0.3226, -0.2601])
    end
  end

  context "combination of functions" do
    it "add two operation together" do
      y = tf.sin(1.0) + tf.sin(2.0)
      expect(y.eval).to eq(1.7507684116335782)
    end
  end
end