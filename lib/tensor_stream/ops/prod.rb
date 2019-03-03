TensorStream::OpMaker.define_operation :prod do |op|
  op.other_names %w(reduce_prod)
  op.what_it_does "Computes the product of elements across dimensions of a tensor."
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
    input_shape = ts.shape(x)
    y = ts.range(0, ts.rank(x)) if y.nil?
    reduction_indices = ts.reshape(y, [-1])

    output_shape_kept_dims = ts.reduced_shape(input_shape, y)
    tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims)
    grad = ts.reshape(grad, output_shape_kept_dims)
    grad = ts.tile(grad, tile_scaling)

    perm, reduced_num, other_num = ts.device("/cpu:0") {
      rank = ts.rank(x)
      reduction_indices = (reduction_indices + rank) % rank
      reduced = ts.cast(reduction_indices, :int32)
      idx = ts.range(0, rank)
      other, = ts.setdiff1d(idx, reduced)
      [ts.concat([reduced, other], 0),
       ts.reduce_prod(ts.gather(input_shape, reduced)),
       ts.reduce_prod(ts.gather(input_shape, other)),]
    }

    permuted = ts.transpose(x, perm)
    permuted_shape = ts.shape(permuted)

    reshaped = ts.reshape(permuted, [reduced_num, other_num])

    # Calculate product, leaving out the current entry
    left = ts.cumprod(reshaped, axis: 0, exclusive: true)
    right = ts.cumprod(reshaped, axis: 0, exclusive: true, reverse: true)
    y = ts.reshape(left * right, permuted_shape)

    # Invert the transpose and reshape operations.
    # Make sure to set the statically known shape information through a reshape.
    out = grad * ts.transpose(y, ts.invert_permutation(perm))
    [ts.reshape(out, input_shape, name: "prod"), nil]
  end

  op.define_shape do |tensor|
    _infer_reduction_op_shape(tensor)
  end
end