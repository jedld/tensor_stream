TensorStream::OpMaker.define_operation :tanh do |op|
  op.what_it_does "Computes tanh of input element-wise."

  op.parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'
  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    grad * i_op(:tanh_grad, params[0])
  end
end