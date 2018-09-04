RSpec.shared_examples "images ops" do
  extend SupportedOp

  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    tf.reset_default_graph
    sess.clear_session_cache
  end

  supported_op ".decode_png" do
    it "converts png file to a tensor" do
      file_path = File.join('spec','fixtures', 'ruby_16.png')
      decoded_image = tf.image.decode_png(file_path)
      expect(sess.run(decoded_image)).to eq([])
    end
  end
end