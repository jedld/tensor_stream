TensorStream::OpMaker.define_operation :strided_slice do |op|
  what_it_does "Extracts a strided slice of a tensor "
  what_it_does "this op extracts a slice of size `(end-begin)/stride`
  from the given `input_` tensor. Starting at the location specified by `begin`
  the slice continues by adding `stride` to the index until all dimensions are
  not less than `end`.
  Note that a stride can be negative, which causes a reverse slice."

  parameter :input, "A tensor"
  parameter :_begin, "start index"
  parameter :_end, "end index"
  parameter :strides, "end index", :nil
  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    input, b_index, e_index, strides = params
    x = ts.shape(input, out_type: node.inputs[0].data_type)

    _op(:strided_slice_grad, x, b_index, e_index, strides, grad)
  end

  op.define_shape do |tensor|
  end
end