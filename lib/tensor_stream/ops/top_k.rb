TensorStream::OpMaker.define_operation :top_k do |op|
  op.what_it_does "Finds values and indices of the `k` largest entries for the last dimension."

  op.parameter :input, "1-D or higher `Tensor` with last dimension at least `k`."
  op.option :k, "0-D `int32` `Tensor`.  Number of top elements to look for along the last dimension (along each row for matrices)", 1
  op.option :sorted, "If true the resulting `k` elements will be sorted by the values in descending order.", "true"
  op.option :name, "Optional name", :nil

  op.define_shape do |tensor|
    input_shape = tensor.inputs[0].shape.shape.dup
    k = tensor.options[:k]
    input_shape[-1] = k
    input_shape
  end

  op.define_gradient do |grad, node, params|
  end
end