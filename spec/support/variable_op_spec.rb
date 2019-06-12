RSpec.shared_examples "supported variable behaviors" do
  extend SupportedOp
  let(:ts) { TensorStream }
  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    ts.reset_default_graph
  end

  it "can use variables in placeholders" do
    v1 = tf.variable([1.0, 1.0])
    init = tf.global_variables_initializer
    puts "init variables"
    sess.run(init)

    x = Float.placeholder
    f = x + 2
    expect(sess.run(f, feed_dict: { x => v1 })).to eq([3.0, 3.0])
  end
end