RSpec.shared_examples "standard ops evaluator" do
  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    tf.reset_default_graph
  end

  context ".zeros" do
    it "generates a zero tensor" do
      a = tf.zeros([2,2])
      expect(sess.run(a)).to eq([[0.0, 0.0], [0.0, 0.0]])
    end
  end

  context ".ones" do
    it "generates a ones tensor" do
      ones = tf.ones([2,2])
      expect(sess.run(ones)).to eq([[1.0, 1.0], [1.0, 1.0]])
    end
  end

  context "op level seed" do
    it "is able to set an op level seed" do
      a = tf.random_uniform([1], seed: 1)
      sess = tf.session
      expect(sess.run(a)).to eq([0.417022004702574])
      expect(sess.run(a)).to eq([0.7203244934421581])

      sess2 = tf.session
      expect(sess2.run(a)).to eq([0.417022004702574])
      expect(sess2.run(a)).to eq([0.7203244934421581])
    end
  end

  context ".convert_to_tensor" do
    it "converts native types and wraps them in a tensor" do
      op = tf.convert_to_tensor([1,2,3,4])
      expect(op.name).to eq("Const:1")
      expect(op.data_type).to eq(:int32)
      expect(sess.run(op)).to eq([1,2,3,4])
    end
  end

  context ".random_uniform_initializer" do
    it "initializes variables using the random uniform initializer" do
      tf.set_random_seed(1234)
      u = tf.get_variable('v', shape: [], dtype: :float32, initializer: tf.random_uniform_initializer)
      sess.run(tf.global_variables_initializer)
      expect(tr(sess.run(u))).to eq(0.1915)
    end
  end

  context ".assign_add" do
    [ [[],    1.0                      ],
      [[1],   [1.0]                     ],
      [[2],   [1.0, 1.0]                ],
      [[2,2], [[1.0, 1.0], [1.0, 1.0]]  ]
    ].each do |shape, expected|
      context "shape #{shape}" do
        it "adds a value to the current variable" do
          v = TensorStream.get_variable("v", shape: shape, initializer: TensorStream.zeros_initializer)
          assignment = v.assign_add(1)
          TensorStream.global_variables_initializer.run
          expect(sess.run(assignment)).to eq(expected)
        end
      end
    end
  end

  context ".assign" do
    specify "assign should set value" do
      w = TensorStream.variable(rand, name: "weight", initializer: TensorStream.zeros_initializer)
      TensorStream.global_variables_initializer.run
      sess.run(w.assign(2))
      expect(w.read_value).to eq(2)
    end
  end

  context ".greater" do
    it "returns true if a > b" do
      a = tf.constant(2.0)
      b = tf.constant(3.0)
      expect(sess.run(tf.greater(a, b))).to eq(false)
      expect(sess.run(tf.greater(b, a))).to eq(true)
    end

    it "handles rank 1 or higher" do
      a = tf.constant([[1.1, 1.3], [1.3, 1.2]])
      c = a > 0
      expect(sess.run(c)).to eq([[true, true], [true, true]])
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

  context ".argmin" do
    it "Returns the index with the smallest value across axes of a tensor. " do
      a = tf.constant([
        [31, 23,  4, 24, 27, 34],
        [18,  3, 25,  0,  6, 35],
        [28, 14, 33, 22, 20,  8],
        [13, 30, 21, 19,  7,  9],
        [16,  1, 26, 32,  2, 29],
        [17, 12,  5, 11, 10, 15]])

      b = tf.constant([1,2,3,4,5,6])
      expect(sess.run(tf.argmin(a))).to eq([3, 4, 0, 1, 4, 2])
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
      expect(sess.run(tf.argmax(a))).to eq([0, 3, 2, 4, 0, 1])
      expect(sess.run(tf.argmax(a, 1))).to eq([5, 5, 2, 1, 3, 0])
      expect(sess.run(tf.argmax(a, 0))).to eq([0, 3, 2, 4, 0, 1])
      expect(sess.run(tf.argmax(b, 0))).to eq(5)
      expect(sess.run(tf.argmax(b, 0, output_type: :float32))).to eql(5.0)
    end
  end

  context ".add standard" do
    it "adds 2 tensors element-wise" do
      a = tf.constant(1.0)
      b = tf.constant(2.0)
      expect(sess.run(tf.add(a, b))).to eq(3.0)

      a = tf.constant([1.0, 1.1])
      b = tf.constant([2.0, 1.5])
      expect(tr(sess.run(tf.add(a, b)))).to eq([3.0, 2.6])
    end

    specify "supports broadcasting" do
      a = tf.constant([1.0, 1.1])
      b = tf.constant(2.0)
      expect(tr(sess.run(tf.add(a, b)))).to eq([3.0, 3.1])
    end

    specify "supports broadcasting rank > 1" do
      a = tf.constant([[1.0, 1.1],[2.2, 1.2]])
      b = tf.constant([2.0, 2.1])
      expect(tr(sess.run(tf.add(a, b)))).to eq([[3.0, 3.2], [4.2, 3.3]])

      a = tf.constant([[1.0, 1.1],[2.2, 1.2]])
      b = tf.constant([[2.0], [2.1]])
      expect(tr(sess.run(tf.add(a, b)))).to eq([[3.0, 3.1], [4.3, 3.3]])

      a = tf.constant([[1, 2, 3], [4, 5, 6]])
      b = tf.constant([1, 2, 3])
      d = a + b
      expect(sess.run(d)).to eq([[2, 4, 6], [5, 7, 9]])
    end

    specify do
      a = tf.constant([1.0, 1.1])
      b = tf.constant([2.0, 1.5])

      c = tf.constant(2.0)

      sum1 = a + b
      sum2 = sum1 + c
      expect(tr(sess.run(sum2))).to eq([5.0, 4.6])
    end
  end

  context ".sub" do
    let(:a) { tf.constant([1.0, 2.0, 3.0])}
    let(:b) { tf.constant([0.1, 0.2, 0.3])}
    let(:c) { tf.constant(0.1) }
    let(:m) { tf.constant([[1.0, 2.0, 3.0], [2.0, 3.0 ,4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]) }

    it "substracts two arrays" do
      expect(tr(sess.run((a - b)))).to eq([0.9, 1.8, 2.7])
    end

    it "substracts an array and a constant" do
      expect(tr(sess.run((a - c)))).to eq([0.9, 1.9, 2.9])
    end

    it "substracts a matrix and an array" do
      expect(tr(sess.run((m - a)))).to eq([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [4.0, 4.0, 4.0], [7.0, 7.0, 7.0]])
    end

    specify "gradients" do
      expect(sess.run(tf.gradients(a - b, [a,b]))).to eq([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    end
  end

  describe "randomization functions" do
    before do
      tf.set_random_seed(1234)
      @sess = tf.session
    end

    context ".random_normal" do
      [
        [[],    0.5011628459350929],
        [[1],   [0.5011628459350929] ],
        [[2,3], [[0.5011628459350929, 1.301972948852967, -1.621722019401658], [0.6690221526288901, 0.14937983113945622, -0.783723693080629]] ],
      ].each do |shape, expected|
        describe "shape #{shape}" do
          it "generates random normal values" do
            r = tf.random_normal(shape)
            expect(tr(sess.run(r))).to eq(tr(expected))
          end
        end
      end
    end
  end

  context ".div" do
    let(:a) { tf.constant(2.5) }
    let(:b) { tf.constant(3.1) }

    it "divides to tensors" do
      op = a / b
      expect(tr(sess.run(op))).to eq(0.8065)
    end

    it "supports gradients" do
      grad = tf.gradients(a/b, [a,b])
      expect(tr(sess.run(grad))).to eq([0.3226, -0.2601])
    end
  end

  context ".mul" do
    it "performs elementwise multiplication" do
      a = tf.constant([[1, 2, 3], [4, 5, 6]])

      # c = a * 6
      # expect(sess.run(c)).to eq([[6, 12, 18], [24, 30, 36]])

      b = tf.constant([1, 2, 3])
      d = a * b
      expect(sess.run(d)).to eq([[1, 4, 9], [4, 10, 18]])
    end

    it "constant multiplication" do
      a= tf.constant([[1, 2, 3], [4, 5, 6]])
      c = tf.constant(6) * a
      expect(sess.run(a)).to eq([[1, 2, 3], [4, 5, 6]])

      b= tf.constant([1,2,3,4,5,6])
      d= tf.constant(6) * b
      expect(sess.run(d)).to eq([6, 12, 18, 24, 30, 36])
    end

    it "handles two rank 1 tensors" do
      a = tf.constant([7.0, 7.0, 7.0, 7.0, 7.0])
      b = tf.constant([-0.1079, 2.281999999999999, 1.1489, -0.5005000000000001, -3.5218999999999996])
      c = a * b
      expect(tr(sess.run(c))).to eq([-0.7553, 15.974, 8.0423, -3.5035, -24.6533])
    end

    it "handles different rows" do
      a = tf.constant([[1.0, 1.0], [1.0, 1.0]])
      b = tf.constant([[4.0, 4.0]])
      c = a * b
      expect(sess.run(c)).to eq([[4.0, 4.0], [4.0, 4.0]])
    end

    it "different rank multiplication" do
      a = tf.constant([7.0, 7.0, 7.0, 7.0, 7.0])
      b = tf.constant([
        [2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1]])
      c = a * b
      expect(sess.run(c)).to eq([[14.0, 14.0, 14.0, 14.0, 14.0], [7.0, 7.0, 7.0, 7.0, 7.0]])
    end

    specify "broadcasting" do
      a = tf.constant([[1.0, 1.1], [2.0, 1.0], [1.0, 1.1]])
      b = tf.constant([[1.2], [1.1], [0.2]])
      f = a * b
      expect(tr(sess.run(f))).to eq([[1.2, 1.32], [2.2, 1.1], [0.2, 0.22]])
      f = b * a
      expect(tr(sess.run(f))).to eq([[1.2, 1.32], [2.2, 1.1], [0.2, 0.22]])
    end
  end

  context ".matmul" do
    it "performs matrix multiplication" do
      a = tf.constant([1, 2, 3, 4, 5, 6], shape: [2, 3])
      b = tf.constant([7, 8, 9, 10, 11, 12], shape: [3, 2])
      c = tf.matmul(a, b)
      expect(sess.run(c)).to eq([[ 58,  64],
                            [139, 154]])

      c = a.matmul(b)
      expect(sess.run(c)).to eq([[ 58,  64],
      [139, 154]])

      d = tf.matmul(a, b, transpose_a: true, transpose_b: true)
      expect(sess.run(d)).to eq([[39, 49, 59], [54, 68, 82], [69, 87, 105]])
    end

    specify "gradients" do
      a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      b = tf.constant([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [10.0, 11.0, 12.0]])

      y = tf.matmul(a, tf.sin(b))

      expect(tr(sess.run(y))).to eq([[-2.0631, -4.0106, -2.2707], [-3.3563, -7.0425, -4.2538]])

      g = tf.gradients(y, [a, b])

      expect(tr(sess.run(g))).to eq([[[2.0585, -2.0806, -2.0806], [2.0585, -2.0806, -2.0806]], [[3.7695, -0.7275, -4.5557], [-5.8735, 0.031, 5.907], [-7.5516, 0.0398, 7.5947]]])
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
  [:reciprocal, 10.0, [[0.9091, 0.0621], [0.4762, 0.3333]], -100,  [[-0.8264, -0.0039], [-0.2268, -0.1111]]                         ],
  [:sigmoid, 0.525, [[0.7503, 1.0], [0.8909, 0.9526]], 0.2494, [[0.1874, 0.0], [0.0972, 0.0452]]]
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

      expect(tr(sess.run(grad))).to eq(tr(gradient))
      expect(tr(sess.run(grad_2))).to eq(tr(gradient2))
    end
  end
end

context "#broadcast" do
  context "gets compatible shapes for two tensors" do
    specify "scalar vs scalar" do
      expect(instance.broadcast(1.0, 1.0)).to eq([1.0, 1.0])
    end

    specify "1D vs constant" do
      expect(instance.broadcast([1.0, 2.0], 1.0)).to eq([[1.0, 2.0], [1.0, 1.0]])
      expect(instance.broadcast([1.0, 2.0, 1.0], 1.0)).to eq([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    end

    specify "1D vs 1D" do
      expect(instance.broadcast([1.0, 2.0], 1.0)).to eq([[1.0, 2.0], [1.0, 1.0]])
      expect(instance.broadcast([1.0, 2.0, 3.0], [1.0])).to eq([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    end

    specify "2D vs 1D" do
      expect(instance.broadcast([[1.0, 2.0], [1.0, 2.0]], 1.0)).to eq([[[1.0, 2.0], [1.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])
      expect(instance.broadcast([[1.0, 2.0], [1.0, 2.0]], [1.0])).to eq([[[1.0, 2.0], [1.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])
      expect(instance.broadcast([[1.0, 2.0], [1.0, 2.0]], [3.0, 3.1])).to eq([[[1.0, 2.0], [1.0, 2.0]], [[3.0, 3.1], [3.0, 3.1]]])
    end

    specify "2D vs 2D" do
      expect(instance.broadcast([[1.0, 2.0], [1.0, 2.0]], [[1.0], [1.0]])).to eq([[[1.0, 2.0], [1.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])
      expect(instance.broadcast([[1.0, 2.0, 1.1], [1.0, 2.0, 2.2]], [[1.0], [2.0]])).to eq( [[[1.0, 2.0, 1.1], [1.0, 2.0, 2.2]], [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]])
    end
  end
end

context "#broadcast_dimensions" do
  it "can broadcast various tensors in various shapes" do
    a = [1.0]
    expect(instance.broadcast_dimensions(a, [5])).to eq([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    expect(instance.broadcast_dimensions(a, [2, 1])).to eq([[1.0, 1.0], [1.0, 1.0]])
    expect(instance.broadcast_dimensions(a, [3, 1])).to eq([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    a = [[1.0, 2.0]]
    b = [[1.0],[2.0]]
    expect(instance.broadcast_dimensions(a, [3, 0])).to eq([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    expect(instance.broadcast_dimensions(b, [0, 1])).to eq([[1.0, 1.0], [2.0, 2.0]])
    expect(instance.broadcast_dimensions(a, [])).to eq([[1.0, 2.0]])
    expect(instance.broadcast_dimensions(b, [])).to eq([[1.0], [2.0]])
    expect(instance.broadcast_dimensions([1.0], [2, 1])).to eq([[1.0, 1.0], [1.0, 1.0]])
  end
end

context ".shape" do
  it "returns a 1D tensor representing shape of target tensor" do
    t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
    shape = tf.shape(t)
    expect(sess.run(shape)).to eq([2, 2, 3])

    u = tf.constant(1)
    shape = tf.shape(u)
    expect(sess.run(shape)).to eq([])

    v = tf.constant([[1,2,3],[4,5,6]])
    shape = tf.shape(v)
    expect(sess.run(shape)).to eq([2 ,3])
  end

  it "can set out_type to return a float" do
    v = tf.constant([[1, 2, 3],[4, 5, 6]])
    shape = tf.shape(v, out_type: :float32)
    expect(sess.run(shape)).to eql([2.0, 3.0])
  end
end

context ".reduce_sum" do
  it "computes the sum of elements across dimensions of a tensor." do
    x = tf.constant([[1, 1, 1], [1, 1, 1]])

    expect(sess.run(tf.reduce_sum(x))).to eq(6)
    expect(sess.run(tf.reduce_sum(x, 0))).to eq([2, 2, 2])
    expect(sess.run(tf.reduce_sum(x, 1))).to eq([3, 3])
    expect(sess.run(tf.reduce_sum(x, 1, keepdims: true))).to eq([[3], [3]])
    expect(sess.run(tf.reduce_sum(x, [0, 1]))).to eq(6)

    expect(sess.run(tf.reduce_sum(x, []))).to eq([[1, 1, 1], [1, 1, 1]]) # no reduction
    expect(sess.run(tf.reduce_sum([[1, 1], [1, 1], [1, 1]]))).to eq(6)
  end

  it "negative axis" do
    x = tf.constant([[1, 1, 1], [1, 1, 1]])

    expect(sess.run(tf.reduce_sum(x, -1))).to eq([3, 3])
    expect(sess.run(tf.reduce_sum(x, -2))).to eq([2, 2, 2])
  end

  it "rank > 2 tensor" do
    x = tf.constant([ [[1,1], [1,1]], [[1,1], [1,1]]])
    expect(sess.run(tf.reduce_sum(x))).to eq(8)
    expect(sess.run(tf.reduce_sum(x, [1, 0]))).to eq([4, 4])
    expect(sess.run(tf.reduce_sum(x, 0))).to eq([[2, 2],[2, 2]])

    y = tf.constant([[1.0, 2.0], [0.4, 4.1], [0.2, 4.2]])
    expect(tr(sess.run(tf.reduce_sum(y, [1], keepdims: true)))).to eq([[3.0], [4.5], [4.4]])
  end

  specify "computes the gradients properly" do
    a = tf.constant([[1,2,3],[4,5,6]])
    op = tf.reduce_sum(a)
    expect(sess.run(tf.gradients(op,[a]))).to eq([[[1, 1, 1], [1, 1, 1]]])
  end
end

context ".reduce_prod" do
  it "computes the sum of elements across dimensions of a tensor." do
    x = tf.constant([[2, 1, 2], [2, 1, 2]])
    expect(sess.run(tf.reduce_prod(x))).to eq(16)
    expect(sess.run(tf.reduce_prod(x, 0))).to eq([4, 1, 4])
    expect(sess.run(tf.reduce_prod(x, 1))).to eq([4, 4])
    expect(sess.run(tf.reduce_prod(x, 1, keepdims: true))).to eq([[4], [4]])
    expect(sess.run(tf.reduce_prod(x, [0, 1]))).to eq(16)
  end

  it "reduceing an empty array" do
    x = tf.constant([])
    y = tf.constant([[], []])
    expect(sess.run(tf.reduce_prod(x))).to eq(1.0)
    expect(sess.run(tf.reduce_prod(y, 0))).to eq([])
    expect(sess.run(tf.reduce_prod(y, 1))).to eq([1.0, 1.0])
  end

  xspecify "computes the gradients properly" do
    a = tf.constant([[1,2,3],[4,5,6]])
    op = tf.reduce_prod(a)
    expect(sess.run(tf.gradients(op,[a]))).to eq([[720, 360, 240],[180, 144, 120]])
  end
end

context ".zeros_like" do
  it "generates a zero tensor based on another tensor" do
    a = tf.zeros_like([2,2,2,2,2])
    b = tf.zeros_like([[2,2],[3,3]])
    expect(sess.run(a)).to eq([0, 0, 0, 0, 0])
    expect(sess.run(b)).to eq([[0, 0], [0, 0]])
  end
end

context ".ones_like" do
  it "generates a zero tensor based on another tensor" do
    a = tf.ones_like([2, 2, 2, 2, 2])
    b = tf.ones_like([[2, 2],[3, 3]])
    expect(sess.run(a)).to eq([1, 1, 1, 1, 1])
    expect(sess.run(b)).to eq([[1, 1], [1, 1]])
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
      expect(tr(sess.run(func))).to eq(e1)
      grad = tf.gradients(func, [x, y])
      expect(tr(sess.run(grad))).to eq(e2)
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
end