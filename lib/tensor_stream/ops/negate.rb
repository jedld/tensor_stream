TensorStream::OpMaker.define_operation :negate do |op|
  what_it_does "Computes numerical negative value element-wise."

  parameter :input, "tensor X"
  option :name, "Optional name", :nil
  op.other_names :negative

  define_gradient do |grad, node, params|
    -grad
  end

  op.define_shape do |tensor|
    tensor.inputs[0].shape.shape
  end
end