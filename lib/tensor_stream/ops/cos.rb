TensorStream::OpMaker.define_operation :cos do |op|
  op.what_it_does "Computes cos of input element-wise."

  op.parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'
  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    -grad * ts.sin(params[0])
  end
end