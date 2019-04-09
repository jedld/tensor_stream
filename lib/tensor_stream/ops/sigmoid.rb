TensorStream::OpMaker.define_operation :sigmoid do |op|
  what_it_does "Computes sigmoid of x element-wise."

  parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'
  option :name, "Optional name", :nil

  define_gradient do |grad, _node, params|
    i_op(:sigmoid_grad, params[0], grad)
  end
end