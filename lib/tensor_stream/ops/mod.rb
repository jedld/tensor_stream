TensorStream::OpMaker.define_operation :mod do |op|
  what_it_does "Returns element-wise remainder of division."

  parameter :input_a, "tensor X"
  parameter :input_b, "tensor Y"

  apply_data_type_coercion!
  supports_broadcasting!

  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
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