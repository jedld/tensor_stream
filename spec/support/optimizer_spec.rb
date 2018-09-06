RSpec.shared_examples "optimizer evaluator" do
  extend SupportedOp
  let(:ts) { TensorStream }
  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    ts.reset_default_graph
  end

  context "ADAM optimizer" do
    supported_op :apply_adam do
      specify "rank 0" do
        n_samples = 5

        m = ts.variable(0.0, dtype: :float32)
        b = ts.variable(0.0, dtype: :float32)
        global_step = ts.variable(0, trainable: false)
        x = ts.placeholder(:float32)
        y = ts.placeholder(:float32)

        pred = m * x + b

        cost = ((pred - y) ** 2).reduce(:+) / ( 2 * n_samples)

        optimizer = TensorStream::Train::AdamOptimizer.new.minimize(cost, global_step: global_step)

        init = ts.global_variables_initializer()

        sess.run(init)

        expect(m.read_value).to eq(0.0)
        sess.run(optimizer, feed_dict: { x => 6.2, y => 26.3 })
        expect(tr(m.read_value)).to eq(0.001)
        sess.run(optimizer, feed_dict: { x => 6.2, y => 26.3 })
        expect(tr(m.read_value)).to eq(0.002)
        expect(sess.run(global_step)).to eq(2)
      end

      specify "rank 1" do
        n_samples = 5

        m = ts.variable([0.0, 0.0], dtype: :float32)
        b = ts.variable([0.0, 0.0], dtype: :float32)
        global_step = ts.variable(0, trainable: false)
        x = ts.placeholder(:float32)
        y = ts.placeholder(:float32)

        pred = m * x + b

        cost = ((pred - y) ** 2).reduce(:+) / ( 2 * n_samples)

        optimizer = TensorStream::Train::AdamOptimizer.new.minimize(cost, global_step: global_step)

        init = ts.global_variables_initializer()

        sess.run(init)

        expect(m.read_value).to eq([0.0, 0.0])
        sess.run(optimizer, feed_dict: { x => [6.2, 3.5], y => [26.3, 27.5] })
        expect(tr(m.read_value)).to eq([0.001, 0.001])
        sess.run(optimizer, feed_dict: { x => [6.2, 3.5], y => [26.3, 27.5] })
        expect(tr(m.read_value)).to eq([0.002, 0.002])
        sess.run(optimizer, feed_dict: { x => [6.2, 3.5], y => [26.3, 27.5] })
        expect(tr(m.read_value)).to eq([0.003, 0.003])
        expect(sess.run(global_step)).to eq(3)
      end

      specify ".compute_gradients" do
          m = ts.variable([0.0, 0.0], dtype: :float32)
          b = ts.variable([0.0, 0.0], dtype: :float32)
          global_step = ts.variable(0, trainable: false)
          x = ts.placeholder(:float32)
          y = ts.placeholder(:float32)

          pred = m * x + b
          optimizer = TensorStream::Train::AdamOptimizer.new
          expect { optimizer.compute_gradients(pred, var_list: x) }.to raise_error
          expect { optimizer.compute_gradients(pred, var_list: [x]) }.to raise_error
          expect(optimizer.compute_gradients(pred, var_list: [m]).map { |g, v| g.name}).to eq(["gradient_wrt_Variable:0/mul_3:0_grad/reshape_19:0"])
      end
    end
  end

  context "Gradient Descent optimizer" do
    supported_op :apply_gradient_descent do
      specify "rank 0" do
        n_samples = 5

        m = ts.variable(0.0, dtype: :float32)
        b = ts.variable(0.0, dtype: :float32)
        global_step = ts.variable(0, trainable: false)
        x = ts.placeholder(:float32)
        y = ts.placeholder(:float32)

        pred = m * x + b

        cost = ((pred - y) ** 2).reduce(:+) / ( 2 * n_samples)

        optimizer = TensorStream::Train::GradientDescentOptimizer.new(0.01).minimize(cost, global_step: global_step)

        init = ts.global_variables_initializer()

        sess.run(init)

        expect(m.read_value).to eq(0.0)
        sess.run(optimizer, feed_dict: { x => 6.2, y => 26.3 })
        expect(tr(m.read_value)).to eq(0.3261)
        sess.run(optimizer, feed_dict: { x => 6.2, y => 26.3 })
        expect(tr(m.read_value)).to eq(0.6265)
        expect(sess.run(global_step)).to eq(2)
      end

      specify "rank 1" do
        n_samples = 5

        m = ts.variable([0.0, 0.0], dtype: :float32)
        b = ts.variable([0.0, 0.0], dtype: :float32)
        global_step = ts.variable(0, trainable: false)
        x = ts.placeholder(:float32)
        y = ts.placeholder(:float32)

        pred = m * x + b

        cost = ((pred - y) ** 2).reduce(:+) / ( 2 * n_samples)

        optimizer = TensorStream::Train::GradientDescentOptimizer.new(0.01).minimize(cost, global_step: global_step)

        init = ts.global_variables_initializer()

        sess.run(init)

        expect(m.read_value).to eq([0.0, 0.0])
        sess.run(optimizer, feed_dict: { x => [6.2, 3.5], y => [26.3, 27.5] })
        expect(tr(m.read_value)).to eq([0.3261, 0.1925])
        sess.run(optimizer, feed_dict: { x => [6.2, 3.5], y => [26.3, 27.5] })
        expect(tr(m.read_value)).to eq([0.6265, 0.3799])
        expect(sess.run(global_step)).to eq(2)
      end

      specify ".compute_gradients" do
        m = ts.variable([0.0, 0.0], dtype: :float32)
        b = ts.variable([0.0, 0.0], dtype: :float32)
        global_step = ts.variable(0, trainable: false)
        x = ts.placeholder(:float32)
        y = ts.placeholder(:float32)

        pred = m * x + b
        optimizer = TensorStream::Train::GradientDescentOptimizer.new(0.01)
        expect { optimizer.compute_gradients(pred, var_list: x) }.to raise_error
        expect { optimizer.compute_gradients(pred, var_list: [x]) }.to raise_error
        expect(optimizer.compute_gradients(pred, var_list: [m]).map { |g, v| g.name}).to eq(["gradient_wrt_Variable:0/mul_3:0_grad/reshape_19:0"])
      end
    end
  end

  context "Momentum optimizer" do
    supported_op :apply_momentum do
      let(:momentum) { 1.0 }
      let(:learning_rate) { 0.01 }

      [true, false].each do |use_nesterov|

        let(:expect_values) do
          # use_nesterov => { rank => expected }
          {
            false => { 0 => 0.3261, 1 => [0.3261, 0.1925] },
            true => { 0 => 0.9526, 1 =>  [0.9526, 0.5724] },
          }
        end

        context "use nesterov = #{use_nesterov}" do
          specify "rank 0" do
            n_samples = 5

            m = ts.variable(0.0, dtype: :float32)
            b = ts.variable(0.0, dtype: :float32)
            global_step = ts.variable(0, trainable: false)
            x = ts.placeholder(:float32)
            y = ts.placeholder(:float32)

            pred = m * x + b

            cost = ((pred - y) ** 2).reduce(:+) / ( 2 * n_samples)

            optimizer = TensorStream::Train::MomentumOptimizer.new(learning_rate, momentum, use_nesterov: use_nesterov).minimize(cost, global_step: global_step)

            init = ts.global_variables_initializer()

            sess.run(init)

            expect(m.read_value).to eq(0.0)
            sess.run(optimizer, feed_dict: { x => 6.2, y => 26.3 })
            sess.run(optimizer, feed_dict: { x => 6.2, y => 26.3 })
            expect(tr(m.read_value)).to eq(expect_values[use_nesterov][0])
            expect(sess.run(global_step)).to eq(2)
          end

          specify "rank 1" do
            n_samples = 5

            m = ts.variable([0.0, 0.0], dtype: :float32)
            b = ts.variable([0.0, 0.0], dtype: :float32)
            global_step = ts.variable(0, trainable: false)
            x = ts.placeholder(:float32)
            y = ts.placeholder(:float32)

            pred = m * x + b

            cost = ((pred - y) ** 2).reduce(:+) / ( 2 * n_samples)

            optimizer = TensorStream::Train::MomentumOptimizer.new(learning_rate, momentum, use_nesterov: use_nesterov).minimize(cost, global_step: global_step)

            init = ts.global_variables_initializer()

            sess.run(init)

            expect(m.read_value).to eq([0.0, 0.0])
            sess.run(optimizer, feed_dict: { x => [6.2, 3.5], y => [26.3, 27.5] })
            sess.run(optimizer, feed_dict: { x => [6.2, 3.5], y => [26.3, 27.5] })
            expect(tr(m.read_value)).to eq(expect_values[use_nesterov][1])
            expect(sess.run(global_step)).to eq(2)
          end
        end

        specify ".compute_gradients" do
          m = ts.variable([0.0, 0.0], dtype: :float32)
          b = ts.variable([0.0, 0.0], dtype: :float32)
          global_step = ts.variable(0, trainable: false)
          x = ts.placeholder(:float32)
          y = ts.placeholder(:float32)
    
          pred = m * x + b
          optimizer = TensorStream::Train::MomentumOptimizer.new(learning_rate, momentum)
          expect { optimizer.compute_gradients(pred, var_list: x) }.to raise_error
          expect { optimizer.compute_gradients(pred, var_list: [x]) }.to raise_error
          expect(optimizer.compute_gradients(pred, var_list: [m]).map { |g, v| g.name}).to eq(["gradient_wrt_Variable:0/mul_3:0_grad/reshape_19:0"])
        end
      end
    end
  end

  context "Adadelta optimizer" do
    supported_op :apply_adadelta do
      specify "rank 0" do
        n_samples = 5

        m = ts.variable(0.0, dtype: :float32)
        b = ts.variable(0.0, dtype: :float32)
        global_step = ts.variable(0, trainable: false)
        x = ts.placeholder(:float32)
        y = ts.placeholder(:float32)

        pred = m * x + b

        cost = ((pred - y) ** 2).reduce(:+) / ( 2 * n_samples)

        optimizer = TensorStream::Train::AdadeltaOptimizer.new.minimize(cost, global_step: global_step)

        init = ts.global_variables_initializer()

        sess.run(init)

        expect(m.read_value).to eq(0.0)
        sess.run(optimizer, feed_dict: { x => 6.2, y => 26.3 })
        sess.run(optimizer, feed_dict: { x => 6.2, y => 26.3 })
        expect(tr(m.read_value,6)).to eq(1.0e-06)
        expect(sess.run(global_step)).to eq(2)
      end

      specify "rank 1" do
        n_samples = 5

        m = ts.variable([0.0, 0.0], dtype: :float32)
        b = ts.variable([0.0, 0.0], dtype: :float32)
        global_step = ts.variable(0, trainable: false)
        x = ts.placeholder(:float32)
        y = ts.placeholder(:float32)

        pred = m * x + b

        cost = ((pred - y) ** 2).reduce(:+) / ( 2 * n_samples)

        optimizer = TensorStream::Train::AdadeltaOptimizer.new.minimize(cost, global_step: global_step)

        init = ts.global_variables_initializer()

        sess.run(init)

        expect(m.read_value).to eq([0.0, 0.0])
        sess.run(optimizer, feed_dict: { x => [6.2, 3.5], y => [26.3, 27.5] })
        sess.run(optimizer, feed_dict: { x => [6.2, 3.5], y => [26.3, 27.5] })
        expect(tr(m.read_value, 6)).to eq([1.0e-06, 1.0e-06])
        expect(sess.run(global_step)).to eq(2)
      end

      specify ".compute_gradients" do
        m = ts.variable([0.0, 0.0], dtype: :float32)
        b = ts.variable([0.0, 0.0], dtype: :float32)
        global_step = ts.variable(0, trainable: false)
        x = ts.placeholder(:float32)
        y = ts.placeholder(:float32)

        pred = m * x + b
        optimizer = TensorStream::Train::AdadeltaOptimizer.new
        expect { optimizer.compute_gradients(pred, var_list: x) }.to raise_error
        expect { optimizer.compute_gradients(pred, var_list: [x]) }.to raise_error
        expect(optimizer.compute_gradients(pred, var_list: [m]).map { |g, v| g.name}).to eq(["gradient_wrt_Variable:0/mul_3:0_grad/reshape_19:0"])
      end
    end
  end
end