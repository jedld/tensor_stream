TensorStream::OpMaker.define_operation :reshape do |op|
  op.what_it_does "Reshapes a tensor."
  op.what_it_does "Given tensor, this operation returns a tensor that has the same values as tensor with shape shape."

  op.parameter :input, "A tensor"
  op.parameter :shape, "A new tensor shape"
  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    [ts.reshape(grad, ts.shape(node.inputs[0])), nil]
  end

  op.define_shape do |tensor|
    new_shape = tensor.inputs[1]&.const_value ? tensor.inputs[1].const_value : nil
    next nil if new_shape.nil?
    next nil if tensor.inputs[0].shape.nil?

    input_shape = tensor.inputs[0].shape.shape
    next new_shape if input_shape.nil? && !new_shape.include?(-1) && !new_shape.include?(nil)
    next nil if input_shape.nil? || input_shape.include?(nil)

    TensorStream::TensorShape.fix_inferred_elements(new_shape, input_shape.reduce(:*))
  end
end