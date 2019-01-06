##
# Tests that deal with model serialization and deserialization
#
RSpec.shared_examples "freezer ops" do
  require 'tensor_stream/utils/freezer'
  let(:ts) { TensorStream }
  let(:freezer) { TensorStream::Freezer.new }
  let(:output_file_location) { '/tmp/lg_model_freezed.yaml' }

  before do
    File.delete(output_file_location) if File.exist?(output_file_location)
  end

  specify "convert a checkpoint to a freezed model" do
    checkpoint_file = File.join('spec', 'fixtures', 'lg_model')

    y1 = freezer.convert(sess, checkpoint_file, output_file_location)
    actual = File.read(output_file_location)
    expected = File.read(File.join('spec', 'fixtures', 'lg_model_freezed.yaml'))
    expect(actual).to eq(expected)

    # check if model works
    target_graph = TensorStream::YamlLoader.new.load_from_file(output_file_location)
    X = target_graph['Placeholder']
    pred = target_graph['add']
    expect(tr(sess.run(pred, feed_dict: { X => 0.2 }))).to eq(0.8514)
  end
end