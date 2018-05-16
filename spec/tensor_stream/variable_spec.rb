require "spec_helper"
require 'benchmark'

RSpec.describe TensorStream::Variable do
  let(:tf) { TensorStream }
  before(:each) do
    described_class.reset_counters
    TensorStream::Operation.reset_counters
    TensorStream::Graph.create_default
  end

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
      session = TensorStream::Session.default_session
      mammal = TensorStream.variable("Elephant", dtype: :string)
      session.run(TensorStream.global_variables_initializer)
      expect(mammal.eval).to eq("Elephant")
    end

    specify "has a default data type" do
      w = TensorStream.variable(rand, name: "weight")
      expect(w.dtype).to eq(:float32)
    end
  end

  context ".get_variable" do  
    let!(:variable) {
      TensorStream.get_variable("other_variable", dtype: TensorStream::Types.int32,
        initializer: TensorStream.constant([23, 42]))
    }

    it "create a variable and add it to the graph" do
      expect(TensorStream.get_collection(TensorStream::GraphKeys::GLOBAL_VARIABLES)).to include(variable)
    end

    it "cannot access variable unless it is initalized" do
      expect { variable.eval }.to raise_exception
    end

    it "can access after initialized" do
      session = TensorStream::Session.default_session
      session.run(TensorStream.global_variables_initializer)
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
      w = tf.get_variable("weight", dtype: :int32, collections: ['test'])
      tf.get_default_graph.get_collection('test').include?(w)
    end
  end

  context ".assign_add" do
    [ [[],    1.0                      ],
      [[1],   [1.0]                     ],
      [[2],   [1.0, 1.0]                ],
      [[2,2], [[1.0, 1.0], [1.0, 1.0]]  ]
    ].each do |shape, expected|
      context "shape #{shape}" do
        it "adds a value to the current variable" do
          session = TensorStream::Session.default_session
          v = TensorStream.get_variable("v", shape: shape, initializer: TensorStream.zeros_initializer)
          assignment = v.assign_add(1)
          TensorStream.global_variables_initializer.run
          expect(session.run(assignment)).to eq(expected)
        end
      end
    end
  end
end