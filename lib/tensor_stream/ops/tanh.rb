TensorStream::OpMaker.define_operation :tanh do |op|
  what_it_does "Computes tanh of input element-wise."

  parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'
  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    grad * i_op(:tanh_grad, params[0])
  end
end