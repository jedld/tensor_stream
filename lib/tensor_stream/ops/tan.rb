TensorStream::OpMaker.define_operation :tan do |op|
  op.what_it_does "Computes tan of input element-wise."

  op.parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'
  op.option :name, "Optional name", :nil

  op.define_gradient do |grad, node, params|
    secx = ts.reciprocal(ts.cos(params[0]))
    secx2 = ts.square(secx)
    grad * secx2
  end
end