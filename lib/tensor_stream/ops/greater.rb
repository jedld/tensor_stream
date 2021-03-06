TensorStream::OpMaker.define_operation :greater do |op|
  op.what_it_does "Returns the truth value of (x > y) element-wise."

  op.parameter :input_a, "tensor X"
  op.parameter :input_b, "tensor Y"

  op.apply_data_type_coercion!
  op.supports_broadcasting!

  op.option :name, "Optional name", :nil

  op.define_data_type do
    :boolean
  end
end