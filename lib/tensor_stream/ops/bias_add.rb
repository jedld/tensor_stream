TensorStream::OpMaker.define_operation :bias_add do |op|
  op.what_it_does "Adds bias to value."

  op.parameter :value, "A Tensor", :nil, validate: 'NUMERIC_TYPES'
  op.parameter :bias, "A 1 D tensor", :nil, validate: 'NUMERIC_TYPES'

  op.supports_broadcasting!
  op.exclude!

  op.option :name, "Optional name", :nil
  op.option :data_format, "A string. 'NHWC' and 'NCHW' are supported.", :nil

  op.define_gradient do |grad, node, _params|
    [grad, _op(:bias_add_grad, grad, data_format: node.options[:data_format])]
  end
end