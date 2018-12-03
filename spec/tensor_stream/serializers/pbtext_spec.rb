require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Pbtext do
  let(:tf) { TensorStream }
  let(:sess) { tf.session }
  let(:serializer) do
    TensorStream::Pbtext.new
  end

  before do
    @fixture = File.read(File.join('spec','fixtures', 'test.pbtxt.proto'))
    tf.reset_default_graph
  end

  xit "saves a graph into pbtext format" do
    # construct a graph
    a = tf.constant([[1.0, 1.1, 2.2, 3.3], [1.1, 2.2, 3.3, 4.0]])
    b = tf.constant(2.0)
    c = tf.variable(1.0, name: "v1")
    d = tf.constant([1,2,3,4])
    e = tf.constant("Hello World")
    f = tf.placeholder(:float32)
    g = tf.constant(["h","e","l","l","o"])
    func = a * b + c
    func2 = tf.reduce_sum(a, [0])

    tf.train.write_graph(tf.get_default_graph, '/tmp', 'ts_test_graph.pbtext')
    expected_content = File.read(File.join('spec', 'fixtures', 'test.pbtxt.proto'))
    test_content = File.read(File.join('/tmp', 'ts_test_graph.pbtext'))
    expect(test_content).to eq(expected_content)
  end
end