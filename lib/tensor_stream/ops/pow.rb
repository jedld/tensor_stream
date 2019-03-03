TensorStream::OpMaker.define_operation :pow do |op|
  op.what_it_does "Computes the power of one value to another X^Y element wise"

  op.parameter :input_a, "tensor X"
  op.parameter :input_b, "tensor Y"

  op.apply_data_type_coercion!
  op.supports_broadcasting!

  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    x, y = params
    z = node
    sx = ts.shape(x)
    sy = ts.shape(y)
    rx, ry = _broadcast_gradient_args(sx, sy)
    gx = ts.reduce_sum(grad * y * ts.pow(x, y - 1), rx)

    log_x = ts.where(x > 0, ts.log(x), ts.zeros_like(x))
    gy = ts.reduce_sum(grad * z * log_x, ry)

    [gx, gy]
  end
end