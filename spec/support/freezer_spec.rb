##
# Tests that deal with model serialization and deserialization
#
RSpec.shared_examples "freezer ops" do
  require "tensor_stream/utils/freezer"
  let(:ts) { TensorStream }
  let(:freezer) { TensorStream::Freezer.new }
  let(:output_file_location) { "/tmp/lg_model_freezed.yaml" }

  before do
    File.delete(output_file_location) if File.exist?(output_file_location)
  end

  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    tf.reset_default_graph
    sess.clear_session_cache
  end

  specify "convert a checkpoint to a freezed model" do
    checkpoint_file = File.join("spec", "fixtures", "lg_model")

    y1 = freezer.convert(sess, checkpoint_file, output_file_location)
    actual = File.read(output_file_location)
    expected = File.read(File.join("spec", "fixtures", "lg_model_freezed.yaml"))
    expect(actual).to eq(expected)

    # check if model works
    target_graph = TensorStream::YamlLoader.new.load_from_file(output_file_location)
    X = target_graph["Placeholder"]
    pred = target_graph["add"]
    expect(tr(sess.run(pred, feed_dict: {X => 0.2}))).to eq(0.8514)
  end

  specify "load a freezed model and eval it" do
    target_graph = TensorStream::YamlLoader.new.load_from_file(File.join("spec", "fixtures", "mnist.yaml"))
    # Load test images
    decoded_image_1 = TensorStream.image.decode_png(File.read(File.join("spec", "fixtures", "0_image.png")), channels: 1)
    decoded_image_2 = TensorStream.image.decode_png(File.read(File.join("spec", "fixtures", "1_image.png")), channels: 1)

    input = target_graph["Placeholder"]
    output = TensorStream.argmax(target_graph["out"], 1)

    reshaped_images = 255.0.t - [decoded_image_1, decoded_image_2].t.cast(:float32)

    result = sess.run(output, feed_dict: {input => reshaped_images})
    expect(result).to eq([7, 2])
  end
end
