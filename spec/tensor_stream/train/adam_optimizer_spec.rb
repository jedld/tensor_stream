require "spec_helper"

RSpec.describe TensorStream::Train::AdamOptimizer do
    let(:ts) { TensorStream }

    [[:ruby_evaluator]].each do |evaluator|
      context "evaluator #{evaluator.join(',')}" do
        let(:sess) { TensorStream.session(evaluator) }

        before(:each) do
            TensorStream::Tensor.reset_counters
            TensorStream::Operation.reset_counters
            ts.reset_default_graph
        end

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
end