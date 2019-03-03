TensorStream::OpMaker.define_operation :case do |op|
  op.exclude!

  op.define_gradient do |grad, node, params|
    n_preds = node.inputs.size - 2

    case_grads = Array.new(n_preds) { |index|
      i_op(:case_grad, index, node.inputs[0], node.inputs[2 + index], grad)
    }

    [nil, i_op(:case_grad, -1, node.inputs[0], node.inputs[1], grad)] + case_grads
  end

  op.define_shape do |tensor|
    tensor.inputs[2]&.shape&.shape
  end
end