TensorStream::OpMaker.define_operation :size do |op|
  op.what_it_does "Returns the size of a tensor."
  op.what_it_does "Returns a 0-D Tensor representing the number of elements in input of type out_type. Defaults to :int32."

  op.parameter :input, "A tensor"
  op.option :name, "Optional name", :nil
  op.option :out_type, "Optional output type", ":int32"

  op.define_gradient do |grad, node, params|
    nil # non differentiable
  end

  op.define_shape do |tensor|
    []
  end
end