TensorStream::OpMaker.define_operation :random_uniform do |op|
  op.what_it_does "Outputs random values from a uniform distribution."

  op.parameter :shape, "A 1-D integer Tensor or array. The shape of the output tensor."

  op.option :name, "Optional name", :nil
  op.option :dtype, "The type of the output: float16, float32, float64, int32, or int64", ":float32"
  op.option :minval, "A 0-D Tensor or ruby value of type dtype. The lower bound on the range of random values to generate. Defaults to 0.", 0
  op.option :maxval, "A 0-D Tensor or ruby value of type dtype. The upper bound on the range of random values to generate. Defaults to 1 if dtype is floating point.", 0
  op.option :seed, " A ruby integer. Used to create a random seed for the distribution. See set_random_seed for behavior.", :nil

  op.define_shape do |tensor|
    a_shape = tensor.inputs[0] ? tensor.inputs[0].const_value : tensor.options[:shape]
    next nil if a_shape.nil?

    a_shape.is_a?(Array) ? a_shape : [a_shape]
  end
end