TensorStream::OpMaker.define_operation :mod do |op|
  op.what_it_does "Returns element-wise remainder of division."

  op.parameter :input_a, "tensor X"
  op.parameter :input_b, "tensor Y"

  op.apply_data_type_coercion!
  op.supports_broadcasting!

  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    x, y = params
    sx = ts.shape(x)
    sy = ts.shape(y)
    rx, ry = _broadcast_gradient_args(sx, sy)
    floor_xy = ts.floor_div(x, y)
    gx = ts.reshape(ts.reduce_sum(grad, rx), sx)
    gy = ts.reshape(ts.reduce_sum(grad * ts.negative(floor_xy), ry), sy)

    [gx, gy]
  end
end