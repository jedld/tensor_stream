TensorStream::OpMaker.define_operation :less do
  what_it_does "Returns the truth value of (x < y) element-wise."

  parameter :input_a, "tensor X"
  parameter :input_b, "tensor Y"

  apply_data_type_coercion!
  supports_broadcasting!

  option :name, "Optional name", :nil

  define_gradient do |grad, node, _params|
    _min_or_max_grad(node.inputs, grad, ->(a, b) { ts.less(a, b) })
  end

  define_data_type do
    :boolean
  end
end