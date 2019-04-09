TensorStream::OpMaker.define_operation :round do |op|
  what_it_does "Rounds the values of a tensor to the nearest integer, element-wise"

  parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'

  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    nil
  end

  op.define_shape do |tensor|
    tensor.inputs[0].shape.shape
  end
end