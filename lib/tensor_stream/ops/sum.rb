TensorStream::OpMaker.define_operation :sum do |op|
  op.other_names %w(reduce_sum)
  what_it_does "Computes the sum of elements across dimensions of a tensor."
  what_it_does "Reduces input_tensor along the dimensions given in axis. Unless keepdims is true, the rank of the"
  what_it_does "tensor is reduced by 1 for each entry in axis. If keepdims is true, the reduced dimensions are"
  what_it_does "retained with length 1."
  what_it_does "If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned."

  parameter :input_a, "tensor X"
  parameter :axis_p, "tensor X", :nil, validate: 'INTEGER_TYPES'

  option :axis, "axis", :nil, exclude: true
  option :name, "Optional name", :nil
  option :keepdims, "If true, retains reduced dimensions with length 1.", :false

  op.add_custom "input_a = TensorStream.convert_to_tensor(input_a)"
  op.add_custom "return input_a if input_a.shape.scalar?"
  op.add_custom "axis_p = axis_p || axis"
  op.add_custom "axis_p = cast_axis(input_a, axis_p)"

  define_gradient do |grad, node, params|
    x, y = params
    _sum_grad(x, y, grad)
  end

  op.define_shape do |tensor|
    _infer_reduction_op_shape(tensor)
  end
end