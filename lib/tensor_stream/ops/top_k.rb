TensorStream::OpMaker.define_operation :top_k do |op|
  what_it_does "Finds values and indices of the `k` largest entries for the last dimension."

  parameter :input, "1-D or higher `Tensor` with last dimension at least `k`."
  parameter :k, "0-D `int32` `Tensor`.  Number of top elements to look for along the last dimension (along each row for matrices)", 1
  option :sorted, "If true the resulting `k` elements will be sorted by the values in descending order.", "true"
  option :name, "Optional name", :nil

  op.add_custom_post "[result[0], result[1]]"

  op.define_shape do |tensor|
    next nil unless tensor.inputs[0].shape.known?

    input_shape = tensor.inputs[0].shape.shape.dup
    k = tensor.options[:k]
    input_shape[-1] = k
    input_shape
  end

  define_gradient do |grad, node, params|
    #TODO
  end
end