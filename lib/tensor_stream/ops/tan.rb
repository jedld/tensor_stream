TensorStream::OpMaker.define_operation :tan do |op|
  what_it_does "Computes tan of input element-wise."

  parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'
  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    secx = ts.reciprocal(ts.cos(params[0]))
    secx2 = ts.square(secx)
    grad * secx2
  end
end