require "spec_helper"

RSpec.describe TensorStream::Graph do
  let(:tf) { TensorStream }
  context ".list_local_devices" do
    specify do
      expect(tf.list_local_devices).to include "job:localhost/ts:ruby:cpu"
    end
  end

  context ".device placement" do
    specify do
      v = tf.device("/device:GPU:1") do
        tf.get_variable("v", shape: [1])
      end
      expect(v.device).to eq("/device:GPU:1")
    end
  end
end