TensorStream::OpMaker.define_operation :round do |op|
  op.what_it_does "Rounds the values of a tensor to the nearest integer, element-wise"

  op.parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'

  op.option :name, "Optional name", :nil

  op.define_shape do |tensor|
    tensor.inputs[0].shape.shape
  end
end