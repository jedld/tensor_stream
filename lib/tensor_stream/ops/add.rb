TensorStream::OpMaker.define_operation :add do
  what_it_does "Returns x + y element-wise."

  parameter :input_a, "tensor X"
  parameter :input_b, "tensor Y"

  apply_data_type_coercion!
  supports_broadcasting!

  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    x, y = params
    next [grad, grad] if shapes_fully_specified_and_equal(x, y)

    sx = ts.shape(x, name: "add/shape_x")
    sy = ts.shape(y, name: "add/shape_y")
    rx, ry = _broadcast_gradient_args(sx, sy)

    [ts.reshape(ts.reduce_sum(grad, rx, name: "add/reduce_sum_x"), sx),
     ts.reshape(ts.reduce_sum(grad, ry, name: "add/reduce_sum_y"), sy),]
  end
end