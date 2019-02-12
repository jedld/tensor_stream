TensorStream::OpMaker.define_operation :range do |op|
  op.what_it_does "Creates a sequence of numbers."
  op.what_it_does "Creates a sequence of numbers that begins at start and extends by increments of delta up to but not including limit."

  op.parameter :start, "Acts as first entry in the range if limit is not nil; otherwise, acts as range limit and first entry defaults to 0.", "0"
  op.parameter :limit, "Upper limit of sequence, exclusive. If nil, defaults to the value of start while the first entry of the range defaults to 0.", "0"
  op.parameter :delta, "Number that increments start. Defaults to 1.", 1

  op.option :name, " A name for the operation. Defaults to \"range\".", "\"range\""
  op.option :dtype, "The type of the elements of the resulting tensor.", :nil
  op.option :output_type, "Output data type defaults to int32", ":int32"

  op.define_gradient do |grad, node, params|
    nil # non differentiable
  end

  op.define_shape do |tensor|
    nil
  end
end