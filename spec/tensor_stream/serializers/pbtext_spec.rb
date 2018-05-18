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
  end

  it "saves a graph into pbtext format" do
    # construct a graph
    a = tf.constant([[1.0, 1.1, 2.2, 3.3], [1.1, 2.2, 3.3, 4.0]])
    b = tf.constant(2)
    c = tf.variable(1.0, name: "v1")
    f = a * b + c
    graph_def = f.graph.as_graph_def()
    expect(graph_def).to eq(@fixture)
  end
end