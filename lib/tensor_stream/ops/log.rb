TensorStream::OpMaker.define_operation :log do |op|
  what_it_does "Computes natural logarithm of x element-wise."

  parameter :input, "tensor X"
  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    grad * TensorStream.reciprocal(params[0])
  end

  op.define_shape do |tensor|
    tensor.inputs[0].shape.shape
  end
end