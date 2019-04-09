TensorStream::OpMaker.define_operation :rsqrt do |op|
  what_it_does "Computes reciprocal of square root of x element-wise."

  parameter :input_a, "tensor X", validate: 'FLOATING_POINT_TYPES'
  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    # Returns -0.5 * grad * conj(y)^3.
    i_op(:rsqrt_grad, node, grad)
  end
end