require "spec_helper"

RSpec.describe TensorStream::Train::GradientDescentOptimizer do
    let(:ts) { TensorStream }

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

        optimizer = TensorStream::Train::GradientDescentOptimizer.new(0.01).minimize(cost, global_step: global_step)

        init = ts.global_variables_initializer()

        sess = ts.session
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

        sess = ts.session
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
        expect(optimizer.compute_gradients(pred, var_list: [m]).map { |g, v| g.name}).to eq(["index_24:0"])
    end
end