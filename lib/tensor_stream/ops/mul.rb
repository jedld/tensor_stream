TensorStream::OpMaker.define_operation :mul do |op|
  what_it_does "Returns x * y element-wise."

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

    [ts.reshape(ts.reduce_sum(ts.mul(grad, y), rx), sx),
     ts.reshape(ts.reduce_sum(ts.mul(x, grad), ry), sy)]
  end
end