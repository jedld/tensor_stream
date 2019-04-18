TensorStream::OpMaker.define_operation :expand_dims do |op|
  what_it_does "Inserts a dimension of 1 into a tensor's shape. "
  what_it_does "Given a tensor input, this operation inserts a dimension of 1 at the dimension index axis of input's shape. The "
  what_it_does "dimension index axis starts at zero; if you specify a negative number for axis it is counted backward from the end."

  parameter :input, "A tensor"
  parameter :axis, "Specifies the dimension index at which to expand the shape of input. Must be in the range [-rank(input) - 1, rank(input)]."
  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    [_reshape_to_input(node, grad), nil]
  end

  op.define_shape do |tensor|
    nil
  end
end