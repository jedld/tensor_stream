TensorStream::OpMaker.define_operation :floor_div do |op|
  what_it_does "Returns element-wise integer divistion."

  parameter :input_a, "tensor X"
  parameter :input_b, "tensor Y"

  apply_data_type_coercion!
  supports_broadcasting!

  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    [nil, nil]
  end
end