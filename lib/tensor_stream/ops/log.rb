TensorStream::OpMaker.define_operation :log do |op|
  op.what_it_does "Computes natural logarithm of x element-wise."

  op.parameter :input, "tensor X"
  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    grad * TensorStream.reciprocal(params[0])
  end

  op.define_shape do |tensor|
    tensor.inputs[0].shape.shape
  end
end