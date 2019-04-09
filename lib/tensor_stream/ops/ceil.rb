TensorStream::OpMaker.define_operation :ceil do |op|
  what_it_does "Returns element-wise smallest integer in not less than x"

  parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'

  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    nil
  end

  define_shape do |tensor|
    tensor.inputs[0].shape.shape
  end
end