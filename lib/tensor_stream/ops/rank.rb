TensorStream::OpMaker.define_operation :rank do |op|
  op.what_it_does "Returns the rank of a tensor"

  op.parameter :input, "A tensor"
  op.option :name, "Optional name", :nil

  op.add_custom "input = convert_to_tensor(input)"
  op.add_custom "return cons(input.shape.ndims) if input.shape.known?"

  op.define_shape do |tensor|
    []
  end
end