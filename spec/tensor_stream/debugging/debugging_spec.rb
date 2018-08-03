require 'spec_helper'

RSpec.describe TensorStream::Debugging do
  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    ts.reset_default_graph
    sess.clear_session_cache
  end

  let(:ts) { TensorStream }
  let(:sess) { ts.session }
  context "#add_check_numerics_ops" do
    specify do
      a = ts.constant([1.0, 1.0, 1.0])
      b = ts.constant([0.0, 0.0, 0.0])
      f = a / b + 1

      ts.add_check_numerics_ops
      expect { sess.run(f) }.to raise_error
    end
  end
end