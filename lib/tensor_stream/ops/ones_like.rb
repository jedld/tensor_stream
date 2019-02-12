TensorStream::OpMaker.define_operation :ones_like do |op|
  op.what_it_does "Creates a tensor with all elements set to 1."
  op.what_it_does "Given a single tensor (tensor), this operation returns a"
  op.what_it_does "tensor of the same type and shape as tensor with all elements set to 1."
  op.what_it_does "Optionally, you can specify a new type (dtype) for the returned tensor."


  op.parameter :input, "A tensor"
  op.option :dtype, "Optional new data type to cast into", :nil, alias: :data_type
  op.option :name, "Optional name", :nil

  op.define_shape do |tensor|
    tensor.inputs[0].shape.shape
  end

  op.define_gradient do |grad, node, params|
    nil # non differentiable
  end
end