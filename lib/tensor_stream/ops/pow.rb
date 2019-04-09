TensorStream::OpMaker.define_operation :pow do |op|
  what_it_does "Computes the power of one value to another X^Y element wise"

  parameter :input_a, "tensor X"
  parameter :input_b, "tensor Y"

  apply_data_type_coercion!
  supports_broadcasting!

  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
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