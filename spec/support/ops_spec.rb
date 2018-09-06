RSpec.shared_examples "standard ops evaluator" do
  extend SupportedOp

  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    tf.reset_default_graph
    sess.clear_session_cache
  end

  context ".control_dependencies" do
    it "control inputs must be fully evaluated before executing block" do
      # We define our Variables and placeholders
      x = tf.placeholder(:int32, shape: [], name: 'x')
      y = tf.variable(2, dtype: :int32)

      # We set our assign op
      assign_op = tf.assign(y, y + 1)

      # We build our multiplication (this could be a more complicated graph)
      out = tf.control_dependencies([assign_op]) do
        x * y
      end

      tf.session do |sess|
        sess.run(tf.global_variables_initializer)

        result = 3.times.collect do |i|
          sess.run(out, feed_dict: {x => 1})
        end
        expect(result).to eq([2, 3, 4])
      end
    end
  end

  context ".softmax_cross_entropy_with_logits" do
    it "Computes softmax cross entropy between logits and labels" do
      labels = tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
      outputs = tf.constant([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [1.1, 2.1, 3.0, 4.0, 1.0, 2.0, 3.0]])
      f = tf.nn.softmax_cross_entropy_with_logits_v2(logits: outputs, labels: labels)
      expect(tr(sess.run(f))).to eq([3.7448, 2.654])
    end

    specify "overflow resistance" do
      x = tf.constant([-2046.4904911315384, 2371.594564592362, -1920.025585664249, 266.06257844862205, 570.1462458227674, 2290.6715733914048, 1396.0319189271745, -2750.277642111798, 1758.5654697551304, 3116.9786057465503])
      label = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
      f = tf.nn.softmax_cross_entropy_with_logits_v2(logits: x, labels: label)
      expect(tr(sess.run(f))).to eq(1358.4131)
    end

    specify "gradients" do
      labels = tf.constant([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      outputs = tf.constant([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
      f = tf.nn.softmax_cross_entropy_with_logits_v2(logits: outputs, labels: labels)
      g = tf.gradients(f, [outputs])
      expect(tr(sess.run(g))).to eq([[-0.9764, 0.0643, 0.1747, 0.4748, 0.0236, 0.0643, 0.1747]])
    end

    specify "gradients overflow resistance" do
      x = tf.constant([-2046.4904911315384, 2371.594564592362, -1920.025585664249, 266.06257844862205, 570.1462458227674, 2290.6715733914048, 1396.0319189271745, -2750.277642111798, 1758.5654697551304, 3116.9786057465503])
      label = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
      f = tf.nn.softmax_cross_entropy_with_logits_v2(logits: x, labels: label)
      h = tf.nn.softmax(x)
      expect(sess.run(h)).to eq([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
      g = tf.gradients(f, [x])
      expect(tr(sess.run(g))).to eq([[ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0,  1.0]])
    end
  end

  context ".log_softmax" do
    specify "computes for the log softmax" do
      logits = tf.constant([-2046.4904911315384, 2371.594564592362, -1920.025585664249, 266.06257844862205, 570.1462458227674, 2290.6715733914048, 1396.0319189271745, -2750.277642111798, 1758.5654697551304, 3116.9786057465503])
      f = tf.nn.log_softmax(logits)
      expect(tr(sess.run(f), 3)).to eq([-5163.469, -745.384, -5037.004, -2850.916, -2546.832, -826.307, -1720.947, -5867.256, -1358.413, 0.0])
    end
  end

  it "performs a linear regression" do
    learning_rate = 0.01
    training_epochs = 2
    display_step = 50
    srand(1234)
    train_X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
    7.042,10.791,5.313,7.997,5.654,9.27,3.1]
    train_Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
    2.827,3.465,1.65,2.904,2.42,2.94,1.3]

    n_samples = train_X.size

    X = TensorStream.placeholder("float")
    Y = TensorStream.placeholder("float")

    # Set model weights
    W = TensorStream.variable(rand, name: "weight")
    b = TensorStream.variable(rand, name: "bias")

    # Construct a linear model
    pred = X * W + b

    # Mean squared error
    cost = TensorStream.reduce_sum(TensorStream.pow(pred - Y, 2.0)) / ( 2.0 * n_samples)

    optimizer = TensorStream::Train::GradientDescentOptimizer.new(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = TensorStream.global_variables_initializer

    expect {
      sess.run(init)
      (0..training_epochs).each do |epoch|
        train_X.zip(train_Y).each do |x,y|
          sess.run(optimizer, feed_dict: {X => x, Y => y})
        end

        if (epoch+1) % display_step == 0
          c = sess.run(cost, feed_dict: {X => train_X, Y => train_Y})
          puts("Epoch:", '%04d' % (epoch+1), "cost=",  c, \
              "W=", sess.run(W), "b=", sess.run(b))
        end
      end
    }.to_not change(cost.graph.nodes, :size)

    puts("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict: { X => train_X, Y => train_Y})
    puts("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    expect(tr(W.read_value)).to eq(0.2524)
    expect(tr(b.read_value)).to eq(0.6314)
  end

  it "can evaluate a tensor" do
    c = tf.constant(1.0)
    expect(sess.run(c)).to eq(1.0)
  end

  it "can evaluate an array of tensors" do
    a = tf.constant(1.0)
    input = [tf.constant([1.0, 1.0]), tf.sin(a)]
    expect(tr(sess.run(*input))).to eq([[1.0, 1.0], 0.8415])
  end

  supported_op ".zeros_like" do
    it "Creates a tensor with all elements set to zero." do
      tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
      z = tf.zeros_like(tensor)
      expect(sess.run(z)).to eq([[0, 0, 0], [0, 0, 0]])
    end
  end

  supported_op ".concat" do
    it "Concatenates tensors along one dimension." do
      t1 = [[1, 2, 3], [4, 5, 6]]
      t2 = [[7, 8, 9], [10, 11, 12]]
      expect(sess.run(tf.concat([t1, t2], 0))).to eq([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
      expect(sess.run(tf.concat([t1, t2], 1))).to eq([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]])
    end

    it "negative axis" do
      t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
      t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
      expect(sess.run(tf.concat([t1, t2], -1))).to eq(
      [[[ 1,  2,  7,  4],
        [ 2,  3,  8,  4]],
       [[ 4,  4,  2, 10],
        [ 5,  3, 15, 11]]])
    end
  end

  supported_op ".reshape" do
    it "Reshapes with dimension of 1" do
      t = tf.constant([[1],[1],[1],[1]])
      expect(sess.run(tf.reshape(t, [4]))).to eq([1, 1, 1, 1])
    end

    it "Reshapes a tensor." do
      t = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      expect(sess.run(tf.reshape(t, [3, 3]))).to eq(
        [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])

      t = [[[1, 1], [2, 2]],
           [[3, 3], [4, 4]]]

      expect(sess.run(tf.reshape(t, [2, 4]))).to eq([[1, 1, 2, 2],
        [3, 3, 4, 4]])
    end

    it "reshape to scalar" do
      t = [7]
      expect(sess.run(tf.reshape(t, []))).to eq(7)

      t = 7
      expect(sess.run(tf.reshape(t, []))).to eq(7)
    end

    it "flattens a tensor" do
      t = [[[1, 1, 1],
            [2, 2, 2]],
          [[3, 3, 3],
          [4, 4, 4]],
          [[5, 5, 5],
          [6, 6, 6]]]
      expect(sess.run(tf.shape(t))).to eq([3, 2, 3])
      expect(sess.run(tf.reshape(t, [-1]))).to eq([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])
      expect(sess.run(tf.reshape(t, [2, -1]))).to eq([[1, 1, 1, 2, 2, 2, 3, 3, 3],
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
        sess.run(tf.reshape(t,[3,2,2]))
      }.to raise_exception

    end

    it "inference" do
      t = [[[1, 1, 1],
            [2, 2, 2]],
            [[3, 3, 3],
            [4, 4, 4]],
            [[5, 5, 5],
            [6, 6, 6]]]

      expect(sess.run(tf.reshape(t, [-1, 9]))).to eq([[1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6]])

      expect(sess.run(tf.reshape(t, [ 2, -1, 3]))).to eq(
        [[[1, 1, 1],
          [2, 2, 2],
          [3, 3, 3]],
          [[4, 4, 4],
          [5, 5, 5],
          [6, 6, 6]]])
    end

    specify do
      expect(tr(sess.run(tf.reshape([ 1.7226844, -0.0204582], [2])))).to eq([1.7227, -0.0205])
    end
  end

  context ".glorot_uniform_initializer" do
    it "initializes variables using the Glorot uniform initializer" do
      tf.set_random_seed(1234)
      u = tf.get_variable('v', shape: [], dtype: :float32)
      v = tf.get_variable('v1', shape: [5], dtype: :float32)
      y = tf.get_variable('v2', shape: [3, 3], dtype: :float32)
      sess.run(tf.global_variables_initializer)
      expect(tr(sess.run(u))).to eq(-1.0686)
      expect(tr(sess.run(v))).to eq([0.2442, -0.1245, 0.5707, 0.56, -0.4548])
      expect(tr(sess.run(y))).to eq([
        [-0.4471, 0.6037, 0.9163],
        [0.7519, -0.2844, 0.002],
        [0.3669, 0.4254, -0.2595]])
    end
  end

  # Outputs random values from a uniform distribution.
  # The generated values follow a uniform distribution in the range [minval, maxval). The lower bound minval is included in the range, while the upper bound maxval is excluded.
  # For floats, the default range is [0, 1). For ints, at least maxval must be specified explicitly.
  # In the integer case, the random integers are slightly biased unless maxval - minval is an exact power of two. The bias is small for values of maxval - minval significantly smaller than the range of the output (either 2**32 or 2**64).
  supported_op ".random_uniform" do
    before do
      tf.set_random_seed(1234)
      @sess = create_session
    end

    [
      [[],    0.1915,       0.383         ],
      [[1],   [0.1915],       [0.383]        ],
      [[2,3], [[0.1915, 0.6221, 0.4377], [0.7854, 0.78, 0.2726]],  [[0.383, 1.2442, 0.8755], [1.5707, 1.56, 0.5452]] ]
    ].each do |shape, expected, range_expected|
      describe "shape #{shape}" do
        it "generates random uniform values" do
          expect(tr(@sess.run(tf.random_uniform(shape)))).to eq(expected)
        end

        specify "with ranges" do
          expect(tr(@sess.run(tf.random_uniform(shape, minval: 0, maxval: 2)))).to eq(range_expected)
        end
      end
    end

    context "shape (3,)" do
      it "Creates an operation to generate a random set of values of the given shape" do
        vec = tf.random_uniform([3])
        expect(tr(@sess.run(vec))).to eq([0.1915, 0.6221, 0.4377])

        #evaluating again generates new values
        expect(tr(@sess.run(vec))).to eq([0.7854, 0.78, 0.2726])
      end
    end

    context "shape (2, 2)" do
      it "Creates an operation to generate a random set of values of the given shape" do
        vec = tf.random_uniform([2,2])
        expect(tr(@sess.run(vec))).to eq([[0.1915, 0.6221], [0.4377, 0.7854]])

        #evaluating again generates new values
        expect(tr(@sess.run(vec))).to eq([[0.78, 0.2726], [0.2765, 0.8019]])
      end
    end
  end

  context ".set_random_seed" do
    it "sets the graph level seed" do
      tf.set_random_seed(1000)
      a = tf.random_uniform([1])
      sess = tf.session
      expect(sess.run(a)).to eq([0.6535895854646095])
      expect(sess.run(a)).to eq([0.11500694312440574])

      sess2 = tf.session
      expect(sess2.run(a)).to eq([0.6535895854646095])
      expect(sess2.run(a)).to eq([0.11500694312440574])
    end
  end

  supported_op ".pad" do
    it "pads a tensor, rank 1" do
      t = tf.constant([1, 2, 3])
      paddings = tf.constant([[1,1]])
      expect(sess.run(tf.pad(t, paddings))).to eq([0, 1, 2, 3, 0])
    end

    it "pads a tensor, rank 2" do
      t = tf.constant([[1, 2, 3], [4, 5, 6]])
      paddings = tf.constant([[1, 1], [2, 2]])

      expect(sess.run(tf.pad(t, paddings, mode: "CONSTANT"))).to eq(
        [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 2, 3, 0, 0],
         [0, 0, 4, 5, 6, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]]
      )

      paddings_2 = tf.constant([[0, 1], [0, 2]])
      expect(sess.run(tf.pad(t, paddings_2, mode: "CONSTANT"))).to eq(
        [
         [1, 2, 3, 0, 0],
         [4, 5, 6, 0, 0],
         [0, 0, 0, 0, 0]
        ]
      )

      paddings_3 = tf.constant([[1, 0], [2, 0]])
      expect(sess.run(tf.pad(t, paddings_3, mode: "CONSTANT"))).to eq(
        [[0, 0, 0, 0, 0],
         [0, 0, 1, 2, 3],
         [0, 0, 4, 5, 6]]
      )
    end
  end

  context ".derivative" do
    it "Creates a derivative graph for a computation" do
      x = tf.placeholder(TensorStream::Types.float32)
      p = tf.pow(x, 3)
      g = tf.gradients(p, [x])
      result = sess.run(g,  feed_dict: { x => 2})
      expect(tr(result)).to eq([12])
      expect(tr(sess.run(p, feed_dict: { x => 2}))).to eq(8)

      # f(x) = (sin x) ^ 3
      # dx = 3(sin x)^2 * cos x
      y = tf.sin(x) ** 3
      derivative_function_y = TensorStream::MathGradients.derivative(y, x)
      expect(tr(derivative_function_y.eval(feed_dict: { x => 1 }))).to eq(1.1477)
    end
  end

  supported_op ".eye" do
    it "creates an identity matrix" do
      tf.program do |tf|
        e = tf.eye(2)
        expect(sess.run(e)).to eq([[1.0, 0.0],[0.0, 1.0]])

        e = tf.eye(3)
        expect(sess.run(e)).to eq([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        e = tf.eye(3, num_columns: 2)
        expect(sess.run(e)).to eq([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      end
    end

    specify "using in matrix multiplication" do
      a = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
      b = tf.constant([[0.1, 0.1], [0.1, 0.1], [0.2, 0.2]])
      m = tf.matmul(a, b)
      expect(tr(sess.run(m))).to eq([[0.9, 0.9], [0.9, 0.9]])

      g = tf.gradients(m, [a])
      expect(tr(sess.run(g))).to eq([[[0.2, 0.2, 0.4], [0.2, 0.2, 0.4]]])

      d_wra = tf.matmul(tf.eye(a.shape[0]), b, transpose_b: true)
      expect(tr(sess.run(d_wra))).to eq([[0.1, 0.1, 0.2], [0.1, 0.1, 0.2]])
    end
  end

  supported_op ".expand_dims" do
    specify do
      t = tf.constant([1, 1])
      expect(sess.run(tf.expand_dims(t, 0))).to eq([[1, 1]])
      expect(sess.run(tf.expand_dims(t, 1))).to eq([[1], [1]])
      t = tf.constant(1)
      expect(sess.run(tf.expand_dims(t, 0))).to eq([1])
      t = tf.constant([[1, 1], [2, 2]])
      expect(sess.run(tf.expand_dims(t, 0))).to eq([[[1, 1], [2, 2]]])
    end
  end

  context ".gradients" do
    it "Constructs symbolic derivatives of sum of ys w.r.t. x in xs." do
      a = tf.constant(0.0)
      b = a * 2
      g = tf.gradients(a + b, [a, b], stop_gradients: [a, b])
      h = tf.gradients(a + b, [a, b])

      expect(sess.run(g)).to eq([1.0, 1.0])
      expect(sess.run(h)).to eq([3.0, 1.0])
    end

    it "using stop gradients" do
      a = tf.stop_gradient(tf.constant(0.0))
      b = tf.stop_gradient(a * 2)
      h = tf.gradients(a + b, [a, b])
      expect(sess.run(a+b)).to eq(0)
      expect((a+b).to_math).to eq("\n (\n  0.0 + \n  \n   (\n    0.0 * 2.0))")
      expect(sess.run(h)).to eq([1.0, 1.0])
    end

    it "computes gradient of sin" do
      var = tf.constant(1.0) # Must be a tf.float32 or tf.float64 variable.
      loss = tf.sin(var) # some_function_of() returns a `Tensor`.
      var_grad = tf.gradients(loss, [var])[0]

      expect(tr(sess.run(var_grad))).to eq(0.5403)
    end
  end

  context ".check_numerics" do
    specify do
      a = tf.constant([[0.0, 0.0, 1.0],[0.0, 1.0, 3.1]])
      c = tf.check_numerics(a, "a")
      expect(tr(sess.run(c))).to eq(tr(sess.run(a)))

      b = tf.constant([[0.0, 0.0, 1.0],[Float::NAN, 1.0, 3.1]])
      d = tf.check_numerics(b, "b")
      expect { sess.run(d) }.to raise_exception TensorStream::InvalidArgumentError
    end
  end

  context ".cond" do
    it "returns a specific tensor function depending on the value of the predicate"  do
      x = tf.constant(2.0)
      y = tf.constant(3.0)
      z = tf.multiply(x, y)

      result = tf.cond(x < y, tf.add(x, z), tf.square(y))
      result2 = tf.cond(x > y, -> { tf.add(x, z) }, -> { tf.square(y) })
      expect(sess.run(result)).to eq(8.0)
      expect(sess.run(result2)).to eq(9.0)
    end

    it "supports gradients" do
      x = tf.constant(2.0)
      y = tf.constant(3.0)
      z = tf.multiply(x, y)

      result = tf.cond(x < y, tf.add(x, z), tf.square(y))
      result2 = tf.cond(x > y, tf.add(x, z), tf.square(y))

      grad1 = tf.gradients(result, [x, y])
      grad2 = tf.gradients(result2, [x, y])

      expect(sess.run(grad1)).to eq([4.0, 2.0])
      expect(sess.run(grad2)).to eq([0.0, 6.0])
    end
  end

  supported_op ".reduce_mean" do
    it "Computes the mean of elements across dimensions of a tensor" do
      x = tf.constant([[1.0, 1.0], [2.0, 2.0]])
      expect(sess.run(tf.reduce_mean(x))).to eq(1.5)
      expect(sess.run(tf.reduce_mean(x, 0))).to eq([1.5, 1.5])
      expect(sess.run(tf.reduce_mean(x, 1))).to eq([1.0, 2.0])

      y = tf.constant([[1.0, 1.0, 1.0], [2.0, 2.0, 3.0], [1.5, -1.1, 1.1]])
      expect(tr(sess.run(tf.reduce_mean(y)))).to eq(1.2778)
      expect(tr(sess.run(tf.reduce_mean(y, 0)))).to eq([1.5, 0.6333, 1.7])
      expect(tr(sess.run(tf.reduce_mean(y, 1)))).to eq([1.0, 2.3333, 0.5])
    end

    it ".computes for the gradient" do
      x = tf.constant([[1.0, 1.0], [2.0, 2.0]])
      f = tf.reduce_mean(x)
      g = tf.gradients(f, [x])
      expect(sess.run(g)).to eq([[[0.25, 0.25], [0.25, 0.25]]])
    end
  end

  supported_op ".tile" do
    it "Constructs a tensor by tiling a given tensor." do
      a = tf.constant([[1, 2, 3, 4], [1, 2, 3, 4]])
      expect(sess.run(tf.tile(a,[1, 0]))).to eq([])
      expect(sess.run(tf.tile(a,[0, 1]))).to eq([])
      expect(sess.run(tf.tile(a,[1, 1]))).to eq([[1, 2, 3, 4], [1, 2, 3, 4]])
      expect(sess.run(tf.tile(a,[2, 1]))).to eq([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
      expect(sess.run(tf.tile(a,[1, 2]))).to eq([[1, 2, 3, 4, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3, 4]])
    end

    specify "gradients" do
      a = tf.constant([[1, 2, 3, 4], [1, 2, 3, 4]])
      op = tf.tile(a, [1, 2])
      g = tf.gradients(op, [a])
      expect(sess.run(g)).to eq([[[2, 2, 2, 2], [2, 2, 2, 2]]])
    end
  end

  supported_op ".stack" do
    specify do
      x = tf.constant([0, 3])
      y = tf.constant([1, 4])
      z = tf.constant([2, 5])
      expect(sess.run(tf.stack([x, y, z]))).to eq([[0, 3], [1, 4], [2, 5]])
      expect(sess.run(tf.stack([x, y, z], axis: 1))).to eq([[0, 1, 2], [3, 4, 5]])
    end

    specify "scalar" do
      x = tf.constant(1)
      y = tf.constant(2)
      z = tf.constant(3)
      expect(sess.run(tf.stack([x,y,z]))).to eq([1, 2, 3])
    end

    specify "rank = 2" do
      x = tf.constant([[0, 1],[2, 3]])
      y = tf.constant([[4, 5],[6, 7]])
      z = tf.constant([[8, 9],[10, 11]])
      expect(sess.run(tf.stack([x, y, z]))).to eq([[[ 0,  1],[ 2,  3]],[[ 4,  5],[ 6,  7]],[[ 8,  9],[10, 11]]]) # [3, 2, 2]
      expect(sess.run(tf.stack([x, y, z], axis: 1))).to eq([[[ 0,  1],[ 4,  5],[ 8,  9]],[[ 2,  3],[ 6,  7],[10, 11]]]) # [2, 3, 2]
      expect(sess.run(tf.stack([x, y, z], axis: 2))).to eq([[[ 0,  4,  8],[ 1,  5,  9]],[[ 2,  6, 10],[ 3,  7, 11]]]) # [2, 2, 3]
      expect(sess.run(tf.stack([x, y, z], axis: -1))).to eq([[[ 0,  4,  8],[ 1,  5,  9]],[[ 2,  6, 10],[ 3,  7, 11]]]) # [2, 2, 3]
    end

    specify "rank = 3" do
      x = tf.constant([[[0, 1],[2, 3]], [[4, 5],[6, 7]]])
      y = tf.constant([[[8, 9],[10, 11]], [[12, 13],[14, 15]]])
      expect(sess.run(tf.stack([x, y]))).to eq([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[8, 9], [10, 11]], [[12, 13], [14, 15]]]])
      expect(sess.run(tf.stack([x, y], axis: 1))).to eq([[[[0, 1], [2, 3]], [[8, 9], [10, 11]]], [[[4, 5], [6, 7]], [[12, 13], [14, 15]]]])
      expect(sess.run(tf.stack([x, y], axis: 2))).to eq([[[[0, 1], [8, 9]], [[2, 3], [10, 11]]], [[[4, 5], [12, 13]], [[6, 7], [14, 15]]]])
      expect(sess.run(tf.stack([x, y], axis: 3))).to eq([[[[0, 8], [1, 9]], [[2, 10], [3, 11]]], [[[4, 12], [5, 13]], [[6, 14], [7, 15]]]])
    end

    xspecify "gradients" do
      x = tf.constant([[0, 1],[2, 3]])
      y = tf.constant([[4, 5],[6, 7]])
      z = tf.constant([[8, 9],[10, 11]])
      f = tf.stack([x, y, z])
      g = tf.gradients(f, [x, y, z])
      expect(sess.run(g)).to eq([])
    end
  end

  context "combination of functions" do
    it "add two operation together" do
      y = tf.sin(1.0) + tf.sin(2.0)
      expect(tr(sess.run(y))).to eq(1.7508)
    end
  end

  supported_op ".max" do
    it "returns the maximum of two tensors" do
      a = tf.constant(1.0)
      b = tf.constant([1.0, 3.0])
      d = tf.constant([3.0, 1.1])
      c = tf.constant(2.1)
      expect(tr(sess.run(tf.max(a,c)))).to eq(2.1)
      expect(sess.run(tf.max(b,d))).to eq([3.0, 3.0])
    end

    it "computes for the gradient" do
      b = tf.constant([1.0, 3.0])
      d = tf.constant([3.0, 1.1])
      g = tf.gradients(tf.max(b,d), [b, d])
      expect(sess.run(g)).to eq([[0.0, 1.0], [1.0, 0.0]])
    end
  end

  supported_op ".min" do
    it "returns the maximum of two tensors" do
      a = tf.constant(1.0)
      b = tf.constant([1.0, 3.0])
      d = tf.constant([3.0, 1.1])
      c = tf.constant(2.1)
      expect(tr(sess.run(tf.min(a,c)))).to eq(1.0)
      expect(tr(sess.run(tf.min(b,d)))).to eq([1.0, 1.1])
    end

    it "computes for the gradient" do
      b = tf.constant([1.0, 3.0])
      d = tf.constant([3.0, 1.1])
      g = tf.gradients(tf.min(b,d), [b, d])
      expect(sess.run(g)).to eq([[1.0, 0.0], [0.0, 1.0]])
    end
  end

  supported_op ".cast" do
    it "converts from one datatype to another" do
      a = tf.constant([1.0, 3.0])
      b = tf.constant([true, true])
      expect(sess.run(tf.cast(a, :int32))).to eql([1, 3])
      expect(sess.run(tf.cast(a, :boolean))).to eql([true, true])
      expect(sess.run(tf.cast(b, :float32))).to eql([1.0, 1.0])
      expect(sess.run(tf.cast(b, :int32))).to eql([1, 1])
    end
  end

  supported_op ".less" do
    it "returns true if a < b" do
      a = tf.constant(2.0)
      b = tf.constant(3.0)
      expect(sess.run(tf.less(a, b))).to eq(true)
      expect(sess.run(tf.less(b, a))).to eq(false)
    end
  end


  supported_op ".greater_equal" do
    it "returns true if a >= b elementwise" do
      a = tf.constant(1.0)
      b = tf.constant(1.0)
      c = tf.constant(2.1)
      d = tf.constant([1.1, 2.1, 3.0])
      e = tf.constant([1.1, 3.1, 1.1])
      expect(sess.run(tf.greater_equal(a,b))).to be
      expect(sess.run(a >= b)).to be
      expect(sess.run(tf.greater_equal(b,c))).to be false
      expect(sess.run(tf.greater_equal(d,e))).to eq([true, false, true])
    end
  end

  supported_op ".less_equal" do
    it "returns true if a >= b elementwise" do
      a = tf.constant(1.0)
      b = tf.constant(1.0)
      c = tf.constant(2.1)
      d = tf.constant([1.1, 2.1, 3.0])
      e = tf.constant([1.1, 3.1, 1.1])
      expect(sess.run(tf.less_equal(a,b))).to be
      expect(a <= b).to be
      expect(sess.run(tf.less_equal(b,c))).to be true
      expect(sess.run(tf.less_equal(d,e))).to eq([true, true, false])
    end
  end

  supported_op ".equal" do
    it "returns the truth value of two tensors" do
      a = tf.constant(1.0)
      b = tf.constant(1.0)
      c = tf.constant(2.1)
      d = tf.constant([[1.0]])
      e = tf.constant([[1.0]])
      f = tf.constant([[2.0]])
      expect(sess.run(tf.equal(a, b))).to eq(true)
      expect(sess.run(tf.equal(a, c))).to eq(false)
      expect(sess.run(tf.equal(d, e))).to eq([[true]])
      expect(sess.run(tf.equal(e, f))).to eq([[false]])

      expect(sess.run(a == b)).to eq(true)
      expect(sess.run(a == c)).to eq(false)
    end
  end


  supported_op ".logical_and" do
    it "Returns the truth value of x AND y element-wise." do
      a = tf.constant([[true, true], [false, true]])
      b = tf.constant([[true, true], [true, true]])
      f = tf.logical_and(a, b)
      expect(sess.run(f)).to eq([[true, true], [false, true]])

      f = a.and(b)
      expect(sess.run(f)).to eq([[true, true], [false, true]])
    end
  end

  supported_op ".not_equal" do
    it "returns the truth value of two tensors" do
      a = tf.constant(1.0)
      b = tf.constant(1.0)
      c = tf.constant(2.1)
      d = tf.constant([[1.0]])
      e = tf.constant([[1.0]])
      f = tf.constant([[2.0]])
      expect(sess.run(tf.not_equal(a, b))).to eq(false)
      expect(sess.run(tf.not_equal(a, c))).to eq(true)
      expect(sess.run(tf.not_equal(d, e))).to eq([[false]])
      expect(sess.run(tf.not_equal(e, f))).to eq([[true]])

      expect(sess.run(a != b)).to eq(false)
      expect(sess.run(a != c)).to eq(true)
    end
  end

  supported_op ".print" do
    it "behaves like identity but prints a message to stdout" do
      x = tf.constant([[2.0, 2.0], [3.0, 3.0]])
      y = tf.print(x, x, message: "this is a prefix")
      z = tf.sin(y)
      expect(tr(sess.run(z))).to eq([[0.9093, 0.9093], [0.1411, 0.1411]])
    end
  end

  supported_op ".slice" do
    it "slices a tensor" do
      t = tf.constant([[[1, 1, 1], [2, 2, 2]],
        [[3, 3, 3], [4, 4, 4]],
        [[5, 5, 5], [6, 6, 6]]])
      expect(sess.run(tf.slice(t, [1, 0, 0], [1, 1, 3]))).to eq([[[3, 3, 3]]])
      expect(sess.run(tf.slice(t, [1, 0, 0], [1, 2, 3]))).to eq([[[3, 3, 3], [4, 4, 4]]])
      expect(sess.run(tf.slice(t, [1, 0, 0], [2, 1, 3]))).to eq([[[3, 3, 3]], [[5, 5, 5]]])
    end

    it "1D tensor slicing" do
      t  = tf.constant([1,2,3,4,5,6,7])
      expect(sess.run(tf.slice(t, [2], [1]))).to eq([3])
    end
  end

  supported_op ".rank" do
    it "returns the rank of a tensor" do
      t1 = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
      t2 = tf.constant(1)
      t3 = tf.constant([1,2])
      rank1 = tf.rank(t1)
      rank2 = tf.rank(t2)
      rank3 = tf.rank(t3)
      expect(sess.run(rank1)).to eq(3)
      expect(sess.run(rank2)).to eq(0)
      expect(sess.run(rank3)).to eq(1)
    end
  end

  supported_op ".negate" do
    it "computes the negative of a tensor" do
      x = tf.constant(0.1)
      y = tf.constant([[1.1, 16.1], [2.1, 3.0]])
      z = -tf.constant(4.1)
      x_negate = tf.negate(x)
      y_negate = tf.negate(y)

      expect(tr(sess.run(x_negate))).to eq(-0.1)
      expect(tr(sess.run(y_negate))).to eq([[-1.1, -16.1], [-2.1, -3.0]])
      expect(tr(sess.run(z))).to eq(-4.1)
    end
  end

  supported_op ".abs" do
    it "Computes the absolute value of a tensor" do
      tf = TensorStream

      a = [[1,2],[-1, 2], [3,-3]]
      b = -1.123

      expect(sess.run(tf.abs(a))).to eq([[1, 2], [1, 2], [3, 3]])
      expect(tr(sess.run(tf.abs(b)))).to eq(1.123)
    end

    specify "should compute for the gradient" do
      a = tf.constant([[1,2],[-1, 2], [3,-3]])
      expect(sess.run(tf.gradients(tf.abs(a),[a]))).to eq([[[ 1,  1],
        [-1,  1],
        [ 1, -1]]])
    end
  end

  supported_op ".sign" do
    it "Returns an element-wise indication of the sign of a number." do
      tf = TensorStream

      a = tf.constant([[1,2],[-1, 2], [3,-3]])
      b = -1.123

      expect(sess.run(tf.sign(a))).to eq([[1, 1], [-1, 1], [1, -1]])
      expect(sess.run(tf.sign(b))).to eq(-1.0)
    end

    specify "gradients" do
      a = tf.constant([[1,2],[-1, 2], [3,-3]])
      g = tf.gradients(tf.sign(a), [a])
      expect(sess.run(g)).to eq([[[0, 0], [0, 0], [0, 0]]])
    end
  end

  supported_op ".transpose" do
    it "transposes matrices" do
      tf.program do |tf|
        x = tf.constant([[1, 2, 3], [4, 5, 6]])
        t = tf.transpose(x)
        expect(sess.run(t)).to eq([[1, 4], [2, 5], [3, 6]])
      end
    end

    specify "perm" do
      x = tf.constant([[1, 2, 3], [4, 5, 6]])
      t = tf.transpose(x, [0, 1])
      expect(sess.run(t)).to eq([[1, 2, 3], [4, 5, 6]])
    end

    specify "multidimensional" do
      x = tf.constant([[[ 0,  1,  2],[ 3,  4,  5]],[[ 6,  7,  8],[9, 10, 11]]])
      op = tf.transpose(x)
      expect(sess.run(op)).to eq([ [[0,6],[3,9]] , [[1,7],[4,10]], [[2,8],[5,11]]])

      op = tf.transpose(x, [0, 2, 1])
      expect(sess.run(op)).to eq([ [[ 0, 3],[ 1,  4],[ 2,  5]], [[ 6,  9],[ 7, 10],[ 8, 11]]] )

      op = tf.transpose(x, [0, 1, 2])
      expect(sess.run(op)).to eq([[[ 0,  1,  2], [ 3,  4,  5]],[[ 6,  7,  8],[ 9, 10, 11]]])
    end

    specify "gradients" do
      x = tf.constant([[1, 2, 3], [4, 5, 6]])
      t = tf.transpose(x)
      g = tf.gradients(t, [x])
      expect(sess.run(g)).to eq([[[1, 1, 1], [1, 1, 1]]])
    end
  end

  supported_op ".zeros" do
    it "generates a zero tensor" do
      a = tf.zeros([2,2])
      expect(sess.run(a)).to eq([[0.0, 0.0], [0.0, 0.0]])
    end
  end

  supported_op ".ones" do
    it "generates a ones tensor" do
      ones = tf.ones([2,2])
      expect(sess.run(ones)).to eq([[1.0, 1.0], [1.0, 1.0]])
    end
  end

  supported_op ".where" do
    it "does an elementwise comparison and picks the appropriate element from x or y" do
      a = tf.constant([1,2,3,4,5])
      b = tf.constant([6,6,6,6,6])
      c = tf.constant([8,8,8,8,8])

      expect(sess.run(tf.where(a > 3, b, c))).to eq([8, 8, 8, 6, 6])
    end

    it "supports gradients" do
      a = tf.constant([1,2,3,4,5])
      b = tf.constant([6,6,6,6,6])
      c = tf.constant([8,8,8,8,8])

      expr = tf.where(a > 3, b, c)
      g = tf.gradients(expr, [b, c])
      expect(sess.run(g)).to eq([[0, 0, 0, 1, 1], [1, 1, 1, 0, 0]])
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

  supported_op ".random_uniform_initializer" do
    it "initializes variables using the random uniform initializer" do
      tf.set_random_seed(1234)
      u = tf.get_variable('v', shape: [], dtype: :float32, initializer: tf.random_uniform_initializer)
      sess.run(tf.global_variables_initializer)
      expect(tr(sess.run(u))).to eq(0.1915)
    end
  end

  supported_op ".zeros_initializer" do
    specify do
      u = tf.get_variable('v', shape: [], dtype: :float32, initializer: tf.zeros_initializer)
      sess.run(tf.global_variables_initializer)
      expect(tr(sess.run(u))).to eq(0.0)
    end
  end

  supported_op ".assign_add" do
    [ [[],    1.0                      ],
      [[1],   [1.0]                     ],
      [[2],   [1.0, 1.0]                ],
      [[2,2], [[1.0, 1.0], [1.0, 1.0]]  ]
    ].each do |shape, expected|
      context "shape #{shape}" do
        it "adds a value to the current variable" do
          v = TensorStream.get_variable("v", shape: shape, initializer: TensorStream.zeros_initializer)
          assignment = v.assign_add(1)
          sess.run(TensorStream.global_variables_initializer)
          expect(sess.run(assignment)).to eq(expected)
        end
      end
    end
  end

  supported_op ".assign" do
    specify "assign should set value" do
      w = TensorStream.variable(rand, name: "weight", initializer: TensorStream.zeros_initializer)
      sess.run(TensorStream.global_variables_initializer)
      sess.run(w.assign(2))
      expect(tr(w.read_value)).to eq(2)
    end
  end

  supported_op ".greater" do
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

  supported_op ".pow" do
    it "Computes the power of tensor x to tensor y" do
      x = tf.constant([[2, 2], [3, 3]])
      y = tf.constant([[8, 15], [2, 3]])
      p = tf.pow(x, y)  # [[256, 65536], [9, 27]]
      expect(sess.run(p)).to eq([[256, 32768], [9, 27]])

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

  supported_op ".gather" do
    context "Gather slices from params axis axis according to indices." do
      specify "scalars" do
        param = tf.constant([1,2,3,4,5,6])
        indexes = tf.constant([5,4,3,2,1,0])
        f = tf.gather(param, indexes)
        expect(sess.run(f)).to eq([6, 5, 4, 3, 2, 1])
      end

      specify "vectors" do
        param = tf.constant([[1,2,3,4,5,6], [7,8,9,10,11,12]])
        indexes = tf.constant([1])
        f = tf.gather(param, indexes)
        expect(sess.run(f)).to eq([[ 7, 8, 9, 10, 11, 12]])
        indexes = tf.constant([0,1])
        f = tf.gather(param, indexes)
        expect(sess.run(f)).to eq([[1,2,3,4,5,6], [7,8,9,10,11,12]])
      end

      specify "matrices" do
        param = tf.constant([[1, 2, 3], [ 4, 5, 6], [7, 8, 9]])
        indexes = tf.constant([ [1, 2]])
        f = tf.gather(param, indexes)
      end
    end
  end

  supported_op ".setdiff1d" do
    specify do
      x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      y = tf.constant([1.0, 3.0, 5.0])
      out, idx = tf.setdiff1d(x, y)
      expect(sess.run(out, idx)).to eq([[2.0, 4.0, 6.0], [1, 3, 5]])
      out, idx = tf.setdiff1d(x, y, index_dtype: :float32)
      expect(sess.run(out, idx)).to eql([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]])
    end
  end

  supported_op ".argmin" do
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

  supported_op '.argmax' do
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

    specify "error checking for axis" do
      x = tf.constant([1,2,3,4,5,6])
      expect {
        sess.run(tf.argmax(x, 1))
      }.to raise_exception TensorStream::InvalidArgumentError
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

    specify "rank 0 vs empty array" do
      a = tf.constant([], dtype: :int32)
      b = tf.constant(0)
      expect(sess.run(a + b)).to eq([])
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

  supported_op ".add_n" do
    specify "adds all inputs elementwise" do
      a = tf.constant([1.0, 2.0, 3.0])
      b = tf.constant([1.1, 2.0, 3.1])
      c = tf.constant([0.0, 2.5, 6.5])

      expect(tr(sess.run(tf.add_n([a])))).to eq([1.0, 2.0, 3.0])
      expect(tr(sess.run(tf.add_n([a, b])))).to eq([2.1, 4.0, 6.1])
      expect(tr(sess.run(tf.add_n([a, b, c])))).to eq([2.1, 6.5, 12.6])
    end

    specify "scalars" do
      a = tf.constant(1.0)
      b = tf.constant(1.1)
      c = tf.constant(2.5)
      expect(tr(sess.run(tf.add_n([a, b, c])))).to eq(4.6)
    end

    specify "gradients" do
      a = tf.constant([1.0, 2.0, 3.0])
      b = tf.constant([1.1, 2.0, 3.1])
      c = tf.constant([0.0, 2.5, 6.5])
      g = tf.gradients(tf.add_n([a, b, c]), [a, b, c])
      expect(sess.run(g)).to eq([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    end
  end

  supported_op ".sub" do
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

  supported_op ".mod" do
    specify do
      a = tf.constant([2, 4])
      b = tf.constant([2, 3])
      f = tf.mod(a, b)
      expect(sess.run(f)).to eq([0, 1])
    end

    specify do
      a = tf.constant([2, 8])
      b = tf.constant([5, 6])
      f = a % b
      expect(sess.run(f)).to eq([2, 2])
    end

    context "gradients" do
      specify do
        a = tf.constant([2, 8])
        b = tf.constant([5, 6])
        f = a % b
        g = tf.gradients(f, [a, b])
        expect(sess.run(g)).to eq([[1, 1], [0, -1]])
      end

      specify do
        a = tf.constant([2.0, 8.0])
        b = tf.constant([5.0, 6.0])
        f = a % b
        g = tf.gradients(f, [a, b])
        expect(sess.run(g)).to eq([[1.0, 1.0], [-0.0, -1.0]])
      end
    end
  end

  supported_op ".floor_div" do
    specify do
      a = tf.constant(2.0)
      b = tf.constant(5.0)
      f = tf.floor_div(a, b)
      expect(sess.run(f)).to eq(0.0)

      a = tf.constant(6.0)
      b = tf.constant(5.0)
      f = tf.floor_div(a, b)
      expect(sess.run(f)).to eq(1.0)
    end
  end

  describe "randomization functions" do
    before do
      tf.set_random_seed(1234)
      @sess = tf.session
    end

    supported_op ".random_standard_normal" do
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

  supported_op ".div" do
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

  supported_op ".mul" do
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
      b = tf.constant([[2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
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

    specify "gradients2" do
      a = tf.constant([[1.1, 2.2], [1.2, 1.5]])
      h2 = tf.constant([[1.0, 5.45], [2.4, 5.6]])
      h3 = tf.constant([[1.1, 2.2], [2.1, 0.6]])
      b2 =  tf.constant([[4.0, 0.5], [0.4, 0.2]])
      out = tf.constant([[2.0, 1.4], [7.1, 1.2]])
      labels = tf.constant([[1.0, 0.0], [1.0, 0.0]])
      layer_2 = tf.add(tf.matmul(a, h2), b2)
      logits = tf.matmul(layer_2, h3) + out
      cross = tf.nn.softmax_cross_entropy_with_logits_v2(logits: logits, labels: labels)
      loss = tf.reduce_mean(cross)
      g = tf.gradients(loss, [h3, h2])
      g2 = tf.gradients(cross, [h3, h2])
      expect(tr(sess.run(cross))).to eq([0.0, 0.0])
      expect(tr(sess.run(loss))).to eq(0.0)
      expect(tr(sess.run(g))).to eq([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    end
  end

  context "math functions" do
    # tests for single parameter algebra functions
    [
      [:sin, 0.0998,   [[0.8912,  0.8632], [0.8632, 0.1411]],  0.995, [[0.4536,  -0.5048], [-0.5048, -0.99]]                      ],
      [:cos, 0.995,    [[0.4536, -0.5048], [-0.5048, -0.99]], -0.0998, [[-0.8912,-0.8632], [-0.8632, -0.1411]]                   ],
      [:tan, 0.1003,   [[1.9648, -1.7098], [-1.7098, -0.1425]], 1.0101,  [[4.8603, 3.9236], [3.9236, 1.0203]]                     ],
      [:tanh, 0.0997,  [[0.8005,  0.9705], [0.9705, 0.9951]],      0.9901, [[0.3592, 0.0582], [0.0582, 0.0099]]                         ],
      [:log, -2.3026,  [[0.0953,  0.7419], [0.7419, 1.0986]],   10.0, [[0.9091, 0.4762], [0.4762, 0.3333]]                        ],
      [:exp, 1.1052,   [[3.0042, 8.1662], [8.1662, 20.0855]], 1.1052, [[3.0042, 8.1662], [8.1662, 20.0855]]          ],
      [:square, 0.01,  [[1.21, 4.41], [4.41, 9.0]],          0.2, [[2.2, 4.2], [4.2, 6.0]]                                    ],
      [:negate, -0.1,  [[-1.1, -2.1], [-2.1, -3.0]],         -1.0, [[-1.0, -1.0], [-1.0, -1.0]]                                 ],
      [:identity, 0.1, [[1.1, 2.1], [2.1, 3.0]],             1.0, [[1, 1], [1, 1]]                                              ],
      [:abs, 0.1,      [[1.1, 2.1], [2.1, 3.0]],             1.0, [[1, 1], [1, 1]]                                              ],
      [:sqrt, 0.3162,  [[1.0488, 1.4491], [1.4491, 1.7321]],   1.5811, [[0.4767,  0.345], [ 0.345, 0.2887]]                       ],
      [:reciprocal, 10.0, [[0.9091,  0.4762], [0.4762, 0.3333]], -100,  [[-0.8264,  -0.2268], [-0.2268, -0.1111]]                         ],
      [:sigmoid, 0.525, [[0.7503, 0.8909], [0.8909, 0.9526]], 0.2494, [[0.1874, 0.0972], [0.0972, 0.0452]]],
      [:floor, 0, [[1, 2], [2, 3]], 0, [[0.0, 0.0], [0.0, 0.0]]],
      [:ceil, 1,[[2, 3], [3, 3]], 0, [[0.0, 0.0], [0.0, 0.0]]]
    ].each do |func, scalar, matrix, gradient, gradient2|
      supported_op ".#{func}" do
        let(:x) { tf.constant(0.1) }
        let(:y) {  tf.constant([[1.1, 2.1], [2.1, 3.0]]) }
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

    [
      [:asin, 0.2014,   [[0.5236, 0.1002], [0.1002, 0.3047]],  1.0206, [[1.1547, 1.005], [1.005, 1.0483]]                      ],
      [:acos, 1.3694,   [[1.0472, 1.4706], [1.4706, 1.2661]],  -1.0206, [[-1.1547, -1.005], [-1.005, -1.0483]]                     ],
      [:atan,  0.1974,  [[0.4636, 0.0997], [0.0997, 0.2915]],  0.9615, [[0.8, 0.9901], [0.9901, 0.9174]]                     ],
    ].each do |func, scalar, matrix, gradient, gradient2|
      supported_op ".#{func}" do
        let(:x) { tf.constant(0.2) }
        let(:y) {  tf.constant([[0.5, 0.1], [0.1, 0.3]]) }
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

  context ".reduced_shape" do
    specify do
      rs = tf.reduced_shape([2, 2], 0)
      expect(sess.run(rs)).to eq([1, 2])
    end

    context ".reduced_shape" do
      include TensorStream::OpHelper
      it "returns the output shape of a tensor after reduction assuing keepdims= true" do
        input = tf.constant([[2,3],[3,4]])
        expect(sess.run(tf.reduced_shape(tf.shape(input), 0))).to eq([1, 2])
      end

      specify do
        expect(sess.run(tf.reduced_shape([2, 3], 0))).to eq([1, 3])
      end
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

  supported_op ".range" do
    it "Creates a sequence of numbers that begins at start and extends by increments of delta up to but not including limit" do
      range = tf.range(3, 18, 3)
      expect(sess.run(range)).to eq([3, 6, 9, 12, 15])
    end

    specify do
      range = tf.range(3, 1, -0.5)
      expect(sess.run(range)).to eq([3, 2.5, 2, 1.5])
    end
  end

  supported_op ".sum" do
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

    specify "rank 0 tensor" do
      c = tf.constant(2.0)
      f = tf.reduce_sum(c)
      g = tf.gradients(f, [c])
      expect(sess.run(g)).to eq([1.0])
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

    specify "alternate notation" do
      a = tf.constant([[1,2,3],[4,5,6]]).reduce(:+)
      expect(sess.run(a)).to eq(21)
    end
  end

  supported_op ".prod" do
    it "computes the product of elements across dimensions of a tensor." do
      x = tf.constant([[2, 1, 1], [3, 1, 1]])

      expect(sess.run(tf.reduce_prod(x))).to eq(6)
      expect(sess.run(tf.reduce_prod(x, 0))).to eq([6, 1, 1])
      expect(sess.run(tf.reduce_prod(x, 1))).to eq([2, 3])
      expect(sess.run(tf.reduce_prod(x, 1, keepdims: true))).to eq([[2], [3]])
      expect(sess.run(tf.reduce_prod(x, [0, 1]))).to eq(6)

      expect(sess.run(tf.reduce_prod(x, []))).to eq([[2, 1, 1], [3, 1, 1]]) # no reduction
      expect(sess.run(tf.reduce_prod([[1, 1], [1, 1], [1, 1]]))).to eq(1)
    end
  end

  supported_op ".squared_difference" do
    it "Returns (x - y)(x - y) element-wise." do
      a = tf.constant([[1,2,3],[4,5,6]])
      b = tf.constant([[2,4,6],[1,2,3]])
      op = tf.squared_difference(a, b)
      expect(sess.run(op)).to eq([[1, 4, 9], [9, 9, 9]])

      a = tf.constant([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
      b = tf.constant(2.0)
      op = tf.squared_difference(a, b)
      expect(sess.run(op)).to eq([[1.0, 0.0, 1.0], [4.0, 9.0, 16.0]])
    end

    it "computes for the gradient" do
      a = tf.constant([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
      b = tf.constant(2.0)
      op = tf.squared_difference(a, b)
      g = tf.gradients(op, [a, b])
      expect(sess.run(g)).to eq([[[-2.0, 0.0, 2.0], [4.0, 6.0, 8.0]], -18.0])
    end
  end

  supported_op ".prod" do
    it "computes the sum of elements across dimensions of a tensor." do
      x = tf.constant([[2, 1, 2], [2, 1, 2]])
      expect(sess.run(tf.reduce_prod(x))).to eq(16)
      expect(sess.run(tf.reduce_prod(x, 0))).to eq([4, 1, 4])
      expect(sess.run(tf.reduce_prod(x, 1))).to eq([4, 4])
      expect(sess.run(tf.reduce_prod(x, 1, keepdims: true))).to eq([[4], [4]])
      expect(sess.run(tf.reduce_prod(x, [0, 1]))).to eq(16)
    end

    xit "reduceing an empty array" do #fails for opencl
      x = tf.constant([])
      y = tf.constant([[], []])
      expect(sess.run(tf.reduce_prod(x))).to eq(1.0)
      expect(sess.run(tf.reduce_prod(y, 0))).to eq([])
      expect(sess.run(tf.reduce_prod(y, 1))).to eq([1.0, 1.0])
    end

    specify "computes the gradients properly" do
      a = tf.constant([[1,2,3],[4,5,6]])
      op = tf.reduce_prod(a)
      expect(sess.run(tf.gradients(op,[a]))).to eq([[[720, 360, 240],[180, 144, 120]]])
    end
  end

  supported_op ".cumprod" do
    let(:x) { tf.constant([2, 3, 4, 5, 6]) }

    specify do
      op = tf.cumprod(x)
      expect(sess.run(op)).to eq([2, 6, 24, 120, 720])
    end

    specify "reverse" do
      op = tf.cumprod(x, reverse: true)
      expect(sess.run(op)).to eq([720, 360, 120, 30, 6])
    end

    specify "exclusive" do
      op = tf.cumprod(x, exclusive: true)
      expect(sess.run(op)).to eq([1, 2, 6, 24, 120])
      op = tf.cumprod(x, exclusive: true, reverse: true)
      expect(sess.run(op)).to eq([360, 120, 30, 6, 1])
    end
  end

  supported_op ".invert_permutation" do
    specify do
      x = tf.constant([3, 4, 0, 2, 1])
      op = tf.invert_permutation(x)
      expect(sess.run(op)).to eq([2, 4, 3, 0, 1])
    end
  end

  supported_op ".zeros_like" do
    it "generates a zero tensor based on another tensor" do
      a = tf.zeros_like([2,2,2,2,2])
      b = tf.zeros_like([[2,2],[3,3]])
      expect(sess.run(a)).to eq([0, 0, 0, 0, 0])
      expect(sess.run(b)).to eq([[0, 0], [0, 0]])
    end
  end

  supported_op ".ones_like" do
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
      supported_op ".#{op}" do


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
      supported_op ".#{op}" do
        specify "mixed rank operation 1  vs 0" do
          func_test(op, a_1, b, expected_1_0, grad_1_0)
        end

        specify "mixed rank operation 2  vs 0" do
          func_test(op, a_2, b, expected_2_0, grad_2_0)
        end
      end
    end
  end

  supported_op ".size" do
    specify do
      t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
      size = tf.size(t)  # 12
      expect(sess.run(size)).to eq(12)
    end
  end

  supported_op ".flow_dynamic_stitch" do
    specify do
      indices = []
      data = []

      indices[0] = 6
      indices[1] = [4, 1]
      indices[2] = [[5, 2], [0, 3]]
      data[0] = [61, 62]
      data[1] = [[41, 42], [11, 12]]
      data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
      expect(sess.run(tf.dynamic_stitch(indices, data))).to eq([[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
        [51, 52], [61, 62]])
    end
  end

  supported_op ".fill" do
    specify do
      g = tf.fill([2, 3], 9)
      expect(sess.run(g)).to eq([[9, 9, 9],
                                 [9, 9, 9]])
    end

    specify do
      g = tf.fill([5, 5], 1.2)
      expect(tr(sess.run(g))).to eq([[1.2, 1.2, 1.2, 1.2, 1.2], [1.2, 1.2, 1.2, 1.2, 1.2], [1.2, 1.2, 1.2, 1.2, 1.2], [1.2, 1.2, 1.2, 1.2, 1.2], [1.2, 1.2, 1.2, 1.2, 1.2]])
    end

    specify do
      g = tf.fill([], 1.0)
      expect(sess.run(g)).to eq(1.0)
    end
  end

  context "nn ops" do
    context ".sigmoid_cross_entropy_with_logits" do
      it "Measures the probability error in discrete classification tasks" do
        labels = tf.constant([[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]])
        outputs = tf.constant([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]])
        f = tf.nn.sigmoid_cross_entropy_with_logits(logits: outputs, labels: labels)
        expect(tr(sess.run(f))).to eq([[0.3133, 2.1269, 0.0486, 4.0181, 0.3133, 0.1269, 3.0486], [0.3133, 2.1269, 0.0486, 4.0181, 0.3133, 0.1269, 3.0486]])
      end

      specify "gradients" do
        labels = tf.constant([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        outputs = tf.constant([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
        f = tf.nn.sigmoid_cross_entropy_with_logits(logits: outputs, labels: labels)
        g = tf.gradients(f, [labels, outputs])
        expect(tr(sess.run(g))).to eq([[-1.0, -2.0, -3.0, -4.0, -1.0, -2.0, -3.0], [-0.2689, 0.8808, -0.0474, 0.982, -0.2689, -0.1192, 0.9526]])
      end
    end

    context ".softmax" do
      it "computes for the softmax of a group of values" do
        outputs = tf.constant([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0],[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]])
        expect(tr(sess.run(tf.nn.softmax(outputs)))).to eq( [[0.0236, 0.0643, 0.1747, 0.4748, 0.0236, 0.0643, 0.1747], [0.0236, 0.0643, 0.1747, 0.4748, 0.0236, 0.0643, 0.1747]])
      end

      specify "rank 1D" do
        outputs = tf.constant([1.0, 1.0, 0.0])
        expect(tr(sess.run(tf.nn.softmax(outputs)))).to eq([0.4223, 0.4223, 0.1554])
      end

      specify "gradients" do
        outputs = tf.constant([1.0, 1.0, 0.0])
        sm = tf.nn.softmax(outputs)
        f = tf.sin(sm)
        g = tf.gradients(f, [outputs])
        result = sess.run(g)
        expect(tr(result,4)).to eq([[-0.005, -0.005, 0.0099]])
      end

      specify "gradients 2D" do
        outputs = tf.constant([[0.1, 0.1, 0.8],[0.2, 0.7, 0.1]])
        sm = tf.nn.softmax(outputs)
        f = tf.sin(sm)
        g = tf.gradients(f, [outputs])
        result = sess.run(g)
        expect(tr(result,4)).to eq([[[0.0115, 0.0115, -0.0231], [0.0082, -0.0173, 0.0092]]])
      end
    end

    context "private ops" do
      context ".broadcast_gradient_args" do
        specify do
          sx, sy = tf.broadcast_gradient_args([], [])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([])
        end

        specify do
          sx, sy = tf.broadcast_gradient_args([2,3], [])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([0, 1])
        end

        it "handles a.shape > b.shape" do
          sx, sy = tf.broadcast_gradient_args([3, 2], [2])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([0])
        end

        it "handles a.shape < b.shape" do
          sx, sy = tf.broadcast_gradient_args([], [2, 2])
          expect(sess.run(sx)).to eq([0, 1])
          expect(sess.run(sy)).to eq([])

          sx, sy = tf.broadcast_gradient_args([2], [3, 2])
          expect(sess.run(sx)).to eq([0])
          expect(sess.run(sy)).to eq([])
        end

        specify do
          sx, sy = tf.broadcast_gradient_args([1], [2])
          expect(sess.run(sx)).to eq([0])
          expect(sess.run(sy)).to eq([])

          sx, sy = tf.broadcast_gradient_args([2], [1])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([0])

          sx, sy = tf.broadcast_gradient_args([3], [1])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([0])


          sx, sy = tf.broadcast_gradient_args([3], [])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([0])

          sx, sy = tf.broadcast_gradient_args([2,2], [])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([0,1])

          sx, sy = tf.broadcast_gradient_args([2,2], [1])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([0,1])

          sx, sy = tf.broadcast_gradient_args([4,4], [1, 1])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([0,1])

          sx, sy = tf.broadcast_gradient_args([4,4], [4, 4])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([])

          sx, sy = tf.broadcast_gradient_args([4,4], [1, 4])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([0])

          sx, sy = tf.broadcast_gradient_args([4,4], [4, 1])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([1])

          sx, sy = tf.broadcast_gradient_args([4,4,4], [1, 4, 4])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([0])

          sx, sy = tf.broadcast_gradient_args([4,4,4], [1, 1, 4])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([0, 1])

          sx, sy = tf.broadcast_gradient_args([4,4,4], [4, 1, 4])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([1])

          sx, sy = tf.broadcast_gradient_args([4,4,4], [4, 4, 1])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([2])

          sx, sy = tf.broadcast_gradient_args([4,4,4], [4, 4])
          expect(sess.run(sx)).to eq([])
          expect(sess.run(sy)).to eq([0])
        end
      end
    end
  end

  context ".squeeze" do
    it "removes dimensions with a size of 1" do
      a = tf.constant([[1],[1],[2],[3]])
      f = tf.squeeze(a)
      expect(sess.run(f)).to eq([1, 1, 2, 3])
    end

    it "squeeze only specific axis" do
      a = tf.constant([[
        [
          [[1],[1],[2],[3]],
          [[1],[1],[2],[3]]
        ]], [[
          [[1],[1],[2],[3]],
          [[1],[1],[2],[3]]
        ]]])
      s = tf.shape(a)
      expect(sess.run(s)).to eq([2, 1, 2, 4, 1])
      f = tf.squeeze(a, axis: 1)
      expect(sess.run(f)).to eq([[[[1], [1], [2], [3]], [[1], [1], [2], [3]]], [[[1], [1], [2], [3]], [[1], [1], [2], [3]]]])
      f = tf.squeeze(a, axis: 4)
      expect(sess.run(f)).to eq([[[[1, 1, 2, 3], [1, 1, 2, 3]]], [[[1, 1, 2, 3], [1, 1, 2, 3]]]])
    end
  end
end