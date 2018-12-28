require "spec_helper"

RSpec.describe TensorStream::ReportTool do
  let(:ts) { TensorStream }
  let(:session) { ts.session(profile_enabled: true) }

  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
    ts.reset_default_graph
    session.clear_session_cache
  end

  context ".profile_for" do
    specify "Generates profile report information" do
      SHAPES = [32, 32]
      a = ts.constant(session.run(ts.random_uniform(SHAPES)))
      b = ts.constant(session.run(ts.random_uniform(SHAPES)))
      a1 = a.dot(b)
      a2 = a1 + b
      a3 = ts.sigmoid(a2)
      a4 = ts.reduce_sum(a3)
      session.run(a4)
      profile = TensorStream::ReportTool.profile_for(session)
      name, _elpased, shape, _source = profile.first
      expect(name).to eq "sum"
      expect(shape).to eq []
    end
  end
end