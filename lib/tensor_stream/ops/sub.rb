TensorStream::OpMaker.define_operation :sub do |op|
  op.other_names %w(subtract)
  op.what_it_does "Returns x - y element-wise."

  op.parameter :input_a, "tensor X"
  op.parameter :input_b, "tensor Y"

  op.apply_data_type_coercion!
  op.supports_broadcasting!

  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    x, y = params
    next [grad, -grad] if shapes_fully_specified_and_equal(x, y)

    sx = ts.shape(x, name: "sub/shape_x")
    sy = ts.shape(y, name: "sub/shape_y")
    rx, ry = _broadcast_gradient_args(sx, sy)

    [ts.reshape(ts.reduce_sum(grad, rx, name: "add/reduce_sub_x"), sx),
     -ts.reshape(ts.reduce_sum(grad, ry, name: "add/reduce_sub_y"), sy),]
  end
end