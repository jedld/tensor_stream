require "spec_helper"
require "benchmark"

RSpec.describe TensorStream::TensorShape do
  context ".infer_shape" do
    let (:shape) { TensorStream::TensorShape }
    it "returns the resulting shape, given two shapes" do
      expect(shape.infer_shape([], [])).to eq([])
      expect(shape.infer_shape([1], [])).to eq([1])
      expect(shape.infer_shape([2, 2], [])).to eq([2, 2])
      expect(shape.infer_shape([nil, 2], [])).to eq([nil, 2])
      expect(shape.infer_shape([nil, 2], [5, 2])).to eq([nil, 2])
      expect(shape.infer_shape([5, 5], [5, 1])).to eq([5, 5])
    end
  end
end
