require "spec_helper"

RSpec.describe TensorStream::Train::LearningRateDecay do
  let(:ts) { TensorStream }


  context ".exponential_decay" do
    specify "Applies exponential decay to the learning rate" do
      n_samples = 5

      m = ts.variable(0.0, dtype: :float32)
      b = ts.variable(0.0, dtype: :float32)
      global_step = ts.variable(0, trainable: false)
      x = ts.placeholder(:float32)
      y = ts.placeholder(:float32)

      pred = m * x + b

      cost = ((pred - y) ** 2).reduce(:+) / ( 2 * n_samples)

      init = ts.global_variables_initializer()
      sess = ts.session
      sess.run(init)

      starter_learning_rate = 0.1
      learning_rate = ts.train.exponential_decay(starter_learning_rate, global_step,
        3, 0.96, staircase: true)

      learning_step = TensorStream::Train::GradientDescentOptimizer.new(learning_rate).minimize(cost, global_step: global_step)
      sess.run(learning_step, feed_dict: { x => 6.2, y => 26.3 })
      expect(sess.run(learning_rate)).to eq(0.1)
      sess.run(learning_step, feed_dict: { x => 6.2, y => 26.3 })
      expect(sess.run(learning_rate)).to eq(0.1)
      sess.run(learning_step, feed_dict: { x => 6.2, y => 26.3 })
      expect(sess.run(learning_rate)).to eq(0.096)
      sess.run(learning_step, feed_dict: { x => 6.2, y => 26.3 })
      expect(sess.run(learning_rate)).to eq(0.096)
      sess.run(learning_step, feed_dict: { x => 6.2, y => 26.3 })
      expect(sess.run(learning_rate)).to eq(0.096)
      sess.run(learning_step, feed_dict: { x => 6.2, y => 26.3 })
      expect(sess.run(learning_rate)).to eq(0.09216)
    end
  end
end