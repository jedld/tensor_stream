TensorStream::OpMaker.define_operation :tile do |op|
  what_it_does "Constructs a tensor by tiling a given tensor."
  what_it_does "This operation creates a new tensor by replicating input multiples times."
  what_it_does "The output tensor's i'th dimension has input.dims(i) * multiples[i] elements,"
  what_it_does "and the values of input are replicated multiples[i] times along the 'i'th dimension. For example, tiling [a b c d] by [2] produces [a b c d a b c d]."

  parameter :input, "A tensor"
  parameter :multiples, "Must be one of the following types: int32, int64. 1-D. Length must be the same as the number of dimensions in input"
  option :name, "Optional name", :nil


  define_gradient do |grad, node, params|
    nil # non differentiable
  end

  op.define_shape do |tensor|
    nil
  end
end