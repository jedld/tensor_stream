TensorStream::OpMaker.define_operation :div do |op|
  op.what_it_does "Returns x / y element-wise."

  op.parameter :input_a, "tensor X"
  op.parameter :input_b, "tensor Y"

  op.apply_data_type_coercion!
  op.supports_broadcasting!

  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    x, y = params
    sx = i_op(:shape, x)
    sy = i_op(:shape, y)
    rx, ry = _broadcast_gradient_args(sx, sy)

    [ts.reshape(ts.reduce_sum(ts.div(grad, y), rx), sx),
     ts.reshape(ts.reduce_sum(grad * ts.div(ts.div(-x, y), y), ry), sy),]
  end
end