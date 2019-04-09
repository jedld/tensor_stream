TensorStream::OpMaker.define_operation :fill do |op|
  what_it_does "This operation creates a tensor of shape dims and fills it with value."

  parameter :dims, "tensor shape"
  parameter :value, "scalar value to fill with"

  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    [nil, TensorStream.reduce_sum(grad)]
  end

  op.define_shape do |tensor|
    a_shape = tensor.inputs[0] ? tensor.inputs[0].const_value : tensor.options[:shape]
    next nil if a_shape.nil?

    a_shape.is_a?(Array) ? a_shape : [a_shape]
  end
end