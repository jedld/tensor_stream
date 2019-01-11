require "spec_helper"

RSpec.describe TensorStream::Train::Utils do
  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    TensorStream.reset_default_graph
  end

  context "#get_global_step" do
    it "gets the current global step variable" do
      expect(TensorStream.train.get_global_step).to be_nil
    end

    it "gets the current global step variable when there is one" do
      TensorStream.train.create_global_step
      expect(TensorStream.train.get_global_step).to be
    end
  end

  context "#create_global_step" do
    it "creates a global step if there is none" do
      global_step_var = TensorStream.train.create_global_step
      expect(global_step_var).to be

      # should be in GLOBAL_STEP collection
      expect(TensorStream.get_collection(TensorStream::GraphKeys::GLOBAL_STEP)).to eq([global_step_var])
    end

    it "returns an error if there is already one" do
      TensorStream.train.create_global_step
      expect {
        TensorStream.train.create_global_step
      }.to raise_error TensorStream::ValueError
    end
  end
end
