require "spec_helper"
require "benchmark"

RSpec.describe TensorStream::Variable do
  let(:tf) { TensorStream }
  before(:each) do
    described_class.reset_counters
    TensorStream::Operation.reset_counters
    tf.reset_default_graph
    TensorStream::Session.default_session.clear_session_cache
  end

  let(:sess) { tf.session(:ruby_evaluator)}

  context "Variable" do
    it "define a variable" do
      # Set model weights
      w = TensorStream.variable(rand, name: "weight")
      expect(TensorStream.get_collection(TensorStream::GraphKeys::GLOBAL_VARIABLES)).to include(w)
    end

    it "can specify initializer" do
      mammal = TensorStream.variable("Elephant", dtype: :string)
      expect { mammal.eval }.to raise_exception
    end

    it "can access after initialized" do
      mammal = TensorStream.variable("Elephant", dtype: :string)
      expect(TensorStream.get_collection(TensorStream::GraphKeys::TRAINABLE_VARIABLES)).to include(mammal)
      sess.run(TensorStream.global_variables_initializer)
      expect(sess.run(mammal)).to eq("Elephant")
    end

    specify "has a default data type" do
      w = TensorStream.variable(rand, name: "weight")
      expect(w.dtype).to eq(:float32)
    end
  end

  context ".get_variable" do
    let!(:variable) {
      tf.get_variable("other_variable", dtype: TensorStream::Types.int32,
        initializer: TensorStream.constant([23, 42]))
    }

    it "create a variable and add it to the graph" do
      expect(TensorStream.get_collection(TensorStream::GraphKeys::GLOBAL_VARIABLES)).to include(variable)
    end

    it "cannot access variable unless it is initalized" do
      expect { variable.eval }.to raise_exception
    end

    it "can access after initialized" do
      sess.run(TensorStream.global_variables_initializer)
      expect(variable.eval).to eq([23, 42])
    end

    it "retrievies an existing variable" do
      w = tf.variable(rand, name: "weight")
      tf.variable_scope("foo", reuse: true) do |scope|
        e = tf.variable(rand, name: "weight")
        expect(e.name).to eq("foo/weight")
        expect(e).to eq(w)
      end
    end

    it "adds to a collection" do
      w = tf.get_variable("weight", dtype: :int32, shape: [5, 5], collections: ["test"])
      tf.get_default_graph.get_collection("test").include?(w)
    end
  end
end
