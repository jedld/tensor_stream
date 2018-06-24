

RSpec.shared_examples "TensorStream::Train::Saver" do
  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    tf.reset_default_graph
    sess.clear_session_cache
  end

  let(:tf) { TensorStream }

  it "saves models using the saver" do
    v1 = tf.get_variable("v1", shape: [3], initializer: tf.zeros_initializer)
    v2 = tf.get_variable("v2", shape: [5], initializer: tf.zeros_initializer)

    inc_v1 = v1.assign(v1+1)
    dec_v2 = v2.assign(v2-1)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer

    # Add ops to save and restore all the variables.
    saver = tf::Train::Saver.new

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.
    sess.run(init_op)
    # Do some work with the model.
    sess.run(inc_v1)
    sess.run(dec_v2)
    # Save the variables to disk.
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)
  end

  it "restores variables using the saver" do
    tf.reset_default_graph()

    # Create some variables.
    v1 = tf.get_variable("v1", shape: [3])
    v2 = tf.get_variable("v2", shape: [5])

    # Add ops to save and restore all the variables.
    saver = tf::Train::Saver.new

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    
    # Restore variables from disk.
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")
    # Check the values of the variables

    expect(sess.run(v1)).to eq([1.0, 1.0, 1.0])
    expect(sess.run(v2)).to eq([-1.0, -1.0, -1.0, -1.0, -1.0])
  end
end