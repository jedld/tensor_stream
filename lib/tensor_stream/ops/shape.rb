TensorStream::OpMaker.define_operation :shape do |op|
  op.what_it_does "This operation returns a 1-D integer tensor representing the shape of input"

  op.parameter :input, "A tensor"
  op.option :name, "Optional name", :nil
  op.option :out_type, "Optional output type", ":int32"

  op.add_custom 'return constant(shape_eval(input, out_type), dtype: out_type, name: "Shape/#{name}") if input.is_a?(Array) && !input[0].is_a?(Tensor)'
  op.add_custom 'return constant(input.shape.shape, dtype: out_type, name: "Shape/#{input.name}_c") if shape_full_specified(input)'

  op.define_shape do |tensor|
    tensor.inputs[0].shape.shape ? [tensor.inputs[0].shape.shape.size] : nil
  end
end