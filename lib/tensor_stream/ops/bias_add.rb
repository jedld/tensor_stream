TensorStream::OpMaker.define_operation :bias_add do |op|
  what_it_does "Adds bias to value."

  parameter :value, "A Tensor", :nil, validate: 'NUMERIC_TYPES'
  parameter :bias, "A 1 D tensor", :nil, validate: 'NUMERIC_TYPES'

  supports_broadcasting!
  exclude!

  option :name, "Optional name", :nil
  option :data_format, "A string. 'NHWC' and 'NCHW' are supported.", :nil

  define_gradient do |grad, node, _params|
    [grad, _op(:bias_add_grad, grad, data_format: node.options[:data_format])]
  end
end