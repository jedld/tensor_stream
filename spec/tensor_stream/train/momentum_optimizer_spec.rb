require "spec_helper"

RSpec.describe TensorStream::Train::MomentumOptimizer do
  let(:ts) { TensorStream }
  let(:momentum) { 1.0 }
  let(:learning_rate) { 0.01 }

  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    ts.reset_default_graph
  end

  [[:ruby_evaluator]].each do |evaluator|

    context "evaluator #{evaluator.join(',')}" do
      let(:sess) { TensorStream.session(evaluator) }
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

            optimizer = described_class.new(learning_rate, momentum, use_nesterov: use_nesterov).minimize(cost, global_step: global_step)

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

            optimizer = described_class.new(learning_rate, momentum, use_nesterov: use_nesterov).minimize(cost, global_step: global_step)

            init = ts.global_variables_initializer()

            sess.run(init)

            expect(m.read_value).to eq([0.0, 0.0])
            sess.run(optimizer, feed_dict: { x => [6.2, 3.5], y => [26.3, 27.5] })
            sess.run(optimizer, feed_dict: { x => [6.2, 3.5], y => [26.3, 27.5] })
            expect(tr(m.read_value)).to eq(expect_values[use_nesterov][1])
            expect(sess.run(global_step)).to eq(2)
          end
        end
      end
    end
  end

  specify ".compute_gradients" do
      m = ts.variable([0.0, 0.0], dtype: :float32)
      b = ts.variable([0.0, 0.0], dtype: :float32)
      global_step = ts.variable(0, trainable: false)
      x = ts.placeholder(:float32)
      y = ts.placeholder(:float32)

      pred = m * x + b
      optimizer = described_class.new(learning_rate, momentum)
      expect { optimizer.compute_gradients(pred, var_list: x) }.to raise_error
      expect { optimizer.compute_gradients(pred, var_list: [x]) }.to raise_error
      expect(optimizer.compute_gradients(pred, var_list: [m]).map { |g, v| g.name}).to eq(["gradient_wrt_Variable:0/mul_3:0_grad/reshape_19:0"])
  end
end