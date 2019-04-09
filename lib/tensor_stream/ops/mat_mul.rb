TensorStream::OpMaker.define_operation :mat_mul do |op|
  op.other_names %w(matmul)
  what_it_does "Multiplies matrix a by matrix b, producing a * b. The inputs must, following any transpositions, be tensors of rank 2 ."

  parameter :input_a, "tensor X"
  parameter :input_b, "tensor Y"

  apply_data_type_coercion!
  supports_broadcasting!

  option :transpose_a, "Transpose matrix A first", :false
  option :transpose_b, "Transpose matrix B first", :false
  option :name, "Optional name", :nil

  define_gradient do |grad, node, params|
    x, y = params
    t_a = node.options[:transpose_a]
    t_b = node.options[:transpose_b]

    if !t_a && !t_b
      grad_a = ts.matmul(grad, y, transpose_b: true)
      grad_b = ts.matmul(x, grad, transpose_a: true)
    elsif !ta && tb
      grad_a = ts.matmul(grad, y)
      grad_b = ts.matmul(grad, x, transpose_a: true)
    elsif t_a && !t_b
      grad_a = ts.matmul(y, grad, transpose_b: true)
      grad_b = ts.matmul(x, grad)
    elsif t_a && t_b
      grad_a = ts.matmul(y, grad, transpose_a: true, transpose_b: true)
      grad_b = ts.matmul(grad, x, transpose_a: true, transpose_b: true)
    end

    [grad_a, grad_b]
  end

  op.define_shape do |tensor|
    next nil if tensor.inputs[0].shape.shape.nil? || tensor.inputs[1].shape.shape.nil?
    next [] if tensor.inputs[0].shape.shape.empty? || tensor.inputs[1].shape.shape.empty?
    next nil if tensor.inputs[0].shape.shape.size != 2 || tensor.inputs[1].shape.shape.size != 2

    shape1, m = if tensor.options[:transpose_a]
      [tensor.inputs[0].shape.shape[0], tensor.inputs[0].shape.shape[1]]
    else
      [tensor.inputs[0].shape.shape[1], tensor.inputs[0].shape.shape[0]]
    end

    shape2, n = if tensor.options[:transpose_b]
      [tensor.inputs[1].shape.shape[1], tensor.inputs[1].shape.shape[0]]
    else
      [tensor.inputs[1].shape.shape[0], tensor.inputs[1].shape.shape[1]]
    end

    next nil if shape1.nil? || shape2.nil? || shape1 < 0 || shape2 < 0

    raise TensorStream::ValueError, "incompatible shape sizes for matrix multiplication (#{shape1} != #{shape2}) #{tensor.inputs[0].shape.shape} vs #{tensor.inputs[1].shape.shape}" if shape1 != shape2

    [m, n]
  end
end