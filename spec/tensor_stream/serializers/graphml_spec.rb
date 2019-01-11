require "spec_helper"
require "benchmark"

RSpec.describe TensorStream::Graphml do
  let(:tf) { TensorStream }
  let(:sess) { tf.session }
  let(:fixture) { File.join("spec", "fixtures", "test.graphml") }
  let(:serializer) do
    TensorStream::Graphml.new
  end

  before do
    @fixture = File.read(fixture)
    tf.reset_default_graph
  end

  xit "saves a graph into graphml format" do
    # construct a graph
    a = tf.constant([[1.0, 1.1, 2.2, 3.3], [1.1, 2.2, 3.3, 4.0]])
    b = tf.constant(2)
    c = tf.variable(1.0, name: "v1")
    d = tf.constant([1, 2, 3, 4])
    e = tf.constant("Hello World")
    f = tf.placeholder(:float32, shape: [2, 2])
    g = tf.constant(["h", "e", "l", "l", "o"])
    func = a * b + c * f
    func2 = tf.reduce_sum(func, [0])
    grad = tf.gradients(func2, [a, d])
    tf.train.write_graph(grad, "/tmp", "ts_test_graph.graphml", serializer: described_class)
    expected_content = File.read(fixture)
    test_content = File.read(File.join("/tmp", "ts_test_graph.graphml"))
    expect(test_content).to eq(expected_content)
  end
end
