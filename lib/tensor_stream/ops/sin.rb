TensorStream::OpMaker.define_operation :sin do |op|
  what_it_does "Computes sin of input element-wise."

  parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'
  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    grad * ts.cos(params[0])
  end
end