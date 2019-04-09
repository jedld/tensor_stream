TensorStream::OpMaker.define_operation :size do |op|
  what_it_does "Returns the size of a tensor."
  what_it_does "Returns a 0-D Tensor representing the number of elements in input of type out_type. Defaults to :int32."

  parameter :input, "A tensor"
  option :name, "Optional name", :nil
  option :out_type, "Optional output type", ":int32"

  define_gradient do |grad, node, params|
    nil # non differentiable
  end

  op.define_shape do |tensor|
    []
  end
end