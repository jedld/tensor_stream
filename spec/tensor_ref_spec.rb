require "spec_helper"
require "tensor_stream/monkey_patches/op_patch"
require "benchmark"

RSpec.describe TensorStream::TensorRef do
  before do
    TensorStream.get_default_graph.reset
  end

  let(:sess) { TensorStream.session }

  specify "Operations cannot be cacheable if there is a TensorRef dependency" do
    a = 1.t
    b = 2.t
    f = a + b
    expect(f.skip_cache).to_not be

    c = TensorStream::TensorRef.new(a)
    f = c + b
    expect(f.skip_cache).to be
    expect(sess.run(f)).to eq(3)

    c.update_ref(b)
    expect(sess.run(f)).to eq(4)
  end
end