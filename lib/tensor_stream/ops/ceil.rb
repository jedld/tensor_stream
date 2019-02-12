TensorStream::OpMaker.define_operation :ceil do |op|
  op.what_it_does "Returns element-wise smallest integer in not less than x"

  op.parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'

  op.option :name, "Optional name", :nil

  op.define_shape do |tensor|
    tensor.inputs[0].shape.shape
  end
end