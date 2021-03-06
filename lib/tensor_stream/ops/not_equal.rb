TensorStream::OpMaker.define_operation :not_equal do |op|
  op.what_it_does "Returns the truth value of (x != y) element-wise."

  op.parameter :input_a, "tensor X"
  op.parameter :input_b, "tensor Y"

  op.apply_data_type_coercion!
  op.supports_broadcasting!

  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    _min_or_max_grad(node.inputs, grad, ->(a, b) { ts.not_equal(a, b) })
  end

  op.define_data_type do
    :boolean
  end
end