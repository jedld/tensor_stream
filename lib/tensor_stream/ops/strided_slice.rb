TensorStream::OpMaker.define_operation :strided_slice do |op|
  op.what_it_does "Extracts a strided slice of a tensor "
  op.what_it_does "this op extracts a slice of size `(end-begin)/stride`
  from the given `input_` tensor. Starting at the location specified by `begin`
  the slice continues by adding `stride` to the index until all dimensions are
  not less than `end`.
  Note that a stride can be negative, which causes a reverse slice."

  op.parameter :input, "A tensor"
  op.parameter :_begin, "start index"
  op.parameter :_end, "end index"
  op.parameter :strides, "end index", :nil
  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
  end

  op.define_shape do |tensor|
  end
end