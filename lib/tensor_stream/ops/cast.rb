TensorStream::OpMaker.define_operation :cast do |op|
  op.exclude!

  op.define_gradient do |grad, node, params|
    t = %i[float16 float32 float64]
    src_type = node.inputs[0].data_type
    dst_type = grad.data_type

    if t.key?(src_type) && t.key?(dst_type)
      next ts.cast(grad, src_type)
    end

    nil
  end
end