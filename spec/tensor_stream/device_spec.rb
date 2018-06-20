require "spec_helper"

RSpec.describe TensorStream::Graph do
  let(:tf) { TensorStream }
  context ".list_local_devices" do
    specify do
      expect(tf.list_local_devices).to include "job:localhost/cpu?evaluator=ruby"
    end
  end
end