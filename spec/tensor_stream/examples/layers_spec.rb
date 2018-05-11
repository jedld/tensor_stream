require "spec_helper"
require 'benchmark'
require 'matrix'

RSpec.describe "create layers sample" do
  xit "creates a dense layers that accepts input vectors" do
    x = TensorStream.placeholder(:float32, shape: [nil, 3])
    linear_model = TensorStream.layers.Dense(units: 1)
    y = linear_model(x)
    puts(y)
    expect(y.to_s).to eq("")
  end
end