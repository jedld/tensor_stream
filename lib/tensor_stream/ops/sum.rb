TensorStream::OpMaker.define_operation :sum do |op|
  op.other_names %w(reduce_sum)
  op.what_it_does "Computes the sum of elements across dimensions of a tensor."
  op.what_it_does "Reduces input_tensor along the dimensions given in axis. Unless keepdims is true, the rank of the"
  op.what_it_does "tensor is reduced by 1 for each entry in axis. If keepdims is true, the reduced dimensions are"
  op.what_it_does "retained with length 1."
  op.what_it_does "If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned."

  op.parameter :input_a, "tensor X"
  op.parameter :axis, "tensor X", :nil, validate: 'INTEGER_TYPES'

  op.option :name, "Optional name", :nil
  op.option :keepdims, "If true, retains reduced dimensions with length 1.", :false

  op.add_custom "input_a = TensorStream.convert_to_tensor(input_a)"
  op.add_custom "return input_a if input_a.shape.scalar?"
  op.add_custom "axis = cast_axis(input_a, axis)"

  op.define_gradient do |grad, node, params|
    x, y = params
    _sum_grad(x, y, grad)
  end

  op.define_shape do |tensor|
    _infer_reduction_op_shape(tensor)
  end
end