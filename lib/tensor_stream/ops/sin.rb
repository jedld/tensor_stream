TensorStream::OpMaker.define_operation :sin do |op|
  op.what_it_does "Computes sin of input element-wise."

  op.parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'
  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    grad * ts.cos(params[0])
  end
end