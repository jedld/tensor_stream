TensorStream::OpMaker.define_operation :floor do |op|
  what_it_does "Returns element-wise largest integer not greater than x."

  parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'

  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    nil
  end

  op.define_shape do |tensor|
    tensor.inputs[0].shape.shape
  end
end