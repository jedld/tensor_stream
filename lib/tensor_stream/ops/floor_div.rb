TensorStream::OpMaker.define_operation :floor_div do |op|
  op.what_it_does "Returns element-wise integer divistion."

  op.parameter :input_a, "tensor X"
  op.parameter :input_b, "tensor Y"

  op.apply_data_type_coercion!
  op.supports_broadcasting!

  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    [nil, nil]
  end
end