TensorStream::OpMaker.define_operation :zeros do |op|
  what_it_does "Creates a tensor with all elements set to zero"

  parameter :shape, "A 1-D integer Tensor or ruby array. The shape of the output tensor."

  option :dtype, "Optional name", ":float32"
  option :name, "Optional name", :nil

  op.define_shape do |tensor|
    a_shape = tensor.inputs[0] ? tensor.inputs[0].const_value : tensor.options[:shape]
    next nil if a_shape.nil?

    a_shape.is_a?(Array) ? a_shape : [a_shape]
  end
end