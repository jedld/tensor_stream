TensorStream::OpMaker.define_operation :less do |op|
  op.what_it_does "Returns the truth value of (x < y) element-wise."

  op.parameter :input_a, "tensor X"
  op.parameter :input_b, "tensor Y"

  op.apply_data_type_coercion!
  op.supports_broadcasting!

  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, _params|
    _min_or_max_grad(node.inputs, grad, ->(a, b) { ts.less(a, b) })
  end

  op.define_data_type do
    :boolean
  end
end