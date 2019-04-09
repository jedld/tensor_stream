TensorStream::OpMaker.define_operation :greater_equal do |op|
  what_it_does "Returns the truth value of (x >= y) element-wise."

  parameter :input_a, "tensor X"
  parameter :input_b, "tensor Y"

  apply_data_type_coercion!
  supports_broadcasting!

  option :name, "Optional name", :nil

  define_data_type do
    :boolean
  end
end