TensorStream::OpMaker.define_operation :ones_like do |op|
  what_it_does "Creates a tensor with all elements set to 1."
  what_it_does "Given a single tensor (tensor), this operation returns a"
  what_it_does "tensor of the same type and shape as tensor with all elements set to 1."
  what_it_does "Optionally, you can specify a new type (dtype) for the returned tensor."


  parameter :input, "A tensor"
  option :dtype, "Optional new data type to cast into", :nil, alias: :data_type
  option :name, "Optional name", :nil

  op.define_shape do |tensor|
    tensor.inputs[0].shape.shape
  end

  define_gradient do |grad, node, params|
    nil # non differentiable
  end
end