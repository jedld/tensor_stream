TensorStream::OpMaker.define_operation :negate do |op|
  op.what_it_does "Computes numerical negative value element-wise."

  op.parameter :input, "tensor X"
  op.option :name, "Optional name", :nil
  op.other_names :negative

  op.define_gradient do |grad, node, params|
    -grad
  end

  op.define_shape do |tensor|
    tensor.inputs[0].shape.shape
  end
end