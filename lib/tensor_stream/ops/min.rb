TensorStream::OpMaker.define_operation :min do |op|
  op.what_it_does "Returns the min of x and y (i.e. x < y ? x : y) element-wise."

  op.parameter :input_a, "tensor X", nil, validate: 'NUMERIC_TYPES'
  op.parameter :input_b, "tensor Y", nil, validate: 'NUMERIC_TYPES'

  op.apply_data_type_coercion!
  op.supports_broadcasting!

  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    _min_or_max_grad(node.inputs, grad, ->(a, b) { ts.less_equal(a, b) })
  end
end