TensorStream::OpMaker.define_operation :max do |op|
  what_it_does "Returns the max of x and y (i.e. x > y ? x : y) element-wise."

  parameter :input_a, "tensor X", nil, validate: 'NUMERIC_TYPES'
  parameter :input_b, "tensor Y", nil, validate: 'NUMERIC_TYPES'

  apply_data_type_coercion!
  supports_broadcasting!

  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    _min_or_max_grad(node.inputs, grad, ->(a, b) { ts.greater_equal(a, b) })
  end
end