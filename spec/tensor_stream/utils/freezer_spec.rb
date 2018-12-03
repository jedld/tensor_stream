require "spec_helper"

require 'tensor_stream/utils/freezer'

RSpec.describe TensorStream::Freezer do
  let(:ts) { TensorStream }
  let(:freezer) { TensorStream::Freezer.new }
  let(:sess) { ts.session }

  before do
    ts.set_random_seed(0)
    x = ts.placeholder(:float32, shape: [nil, 28, 28, 1])
    w1 = ts.variable(ts.truncated_normal([6, 6, 1, 4], stddev: 0.1))
    b1 = ts.variable(ts.ones([4])/10)
    sess = ts.session
    y1 = ts.nn.relu(ts.nn.conv2d(ts.reshape(x, [-1, 28, 28, 1]), w1, [1, 1, 1, 1], 'SAME') + b1)

    # Add ops to save and restore all the variables.
    init = ts.global_variables_initializer
    sess.run(init)
    freezer.save('/tmp/freezed.pbtext', y1)
  end

  xspecify "can read back freezed model" do
    # y1 = freezer.restore('/tmp/freezed.pbtext')
    # result = sess.run(y1, feed_dict: { :x => ts.truncated_normal([1, 28, 28, 1])})
    # expect(result).to eq([])
  end
end