module TensorStream
  # Class that provides auto-differentiation
  class MathGradientsForward
    extend TensorStream::OpHelper

    def self.tf
      TensorStream
    end

    def self.derivative(tensor, wrt_dx, options = {})
      return i_op(:ones_like, tensor) if tensor.equal?(wrt_dx)
      return i_op(:zeros_like, tensor) unless wrt_dx.consumers.include?(tensor.name)

      nodes_to_compute = wrt_dx.consumers.select do |t|
        node = tensor.graph.nodes[t]
        node.consumers.include?(tensor.name) || node.equal?(tensor)
      end.compact + [wrt_dx.name]

      grad = i_op(:ones_like, wrt_dx)

      _propagate(grad, tensor, wrt_dx, nodes_to_compute, options[:stop_gradients])
    end

    def self._propagate(grad, tensor, stop_tensor, nodes_to_compute, stop_gradients = [])
      return grad * i_op(:ones_like, grad) if stop_tensor.equal?(tensor)
      return i_op(:zeros_like, grad) if stop_gradients && _include?(stop_gradients, tensor)
      return i_op(:zeros_like, grad) unless tensor.is_a?(Operation)

      computed_op = if tensor.operation == :matmul
                      s0 =  i_op(:shape, tensor)
                      s1 =  i_op(:shape, grad)
                      identity_0 = i_op(:ones, [s0[0], s1[1]], nil, data_type: tensor.data_type, name: 'matmul/identity0')
                      dx, dy = _compute_derivative(tensor, identity_0)
                      [tf.mul(dx, grad), tf.mul(dy, grad)]
                    elsif _op_supports_broadcast?(tensor)
                      _compute_derivative(tensor, _broadcast_transform(tensor, grad)[1])
                    else
                      _compute_derivative(tensor, grad)
                    end

      if computed_op.is_a?(Array)
        grad_sum = i_op(:zeros_like, grad)
        computed_op.each_with_index do |op_grad, index|
          next if op_grad.nil?

          if nodes_to_compute.include?(tensor.items[index].name)
            grad_sum += _propagate(op_grad, tensor.items[index], stop_tensor, nodes_to_compute, stop_gradients)
          end
        end

        grad_sum
      else
        _propagate(computed_op, tensor.items[0], stop_tensor, nodes_to_compute, stop_gradients)
      end
    end

    def self._forward_propagate(grad, tensor, stop_tensor, nodes_to_compute)
      grad_sum = i_op(:zeros_like, grad)

      tensor.outputs.each do |output|
        next unless nodes_to_compute.include?(output)

        node = tensor.graph.nodes[output]

        computed_op = if node.operation == :matmul
          s0 =  i_op(:shape, node)
          s1 =  i_op(:shape, grad)
          identity_0 = i_op(:ones, [s0[0], s1[1]], nil, data_type: node.data_type, name: 'matmul/identity0')
          dx, dy = _compute_derivative(node, identity_0)
          [tf.mul(dx, grad), tf.mul(dy, grad)]
        elsif _op_supports_broadcast?(node)
          _compute_derivative(node, _broadcast_transform(node, grad)[1])
        else
          _compute_derivative(node, grad)
        end

        next_op = if computed_op.is_a?(Array)
                    grads_input0, grads_input1 = computed_op

                    if node.items[0] && node.items[1] && node.items[0].name == tensor.name &&
                        node.items[1].name == tensor.name &&
                          grads_input0 && grads_input1
                      tf.add(grads_input0, grads_input1, name: 'grad_merge')
                    elsif node.items[0].name == tensor.name
                      grads_input0
                    else
                      grads_input1
                    end
                  else
                    computed_op
                  end

        grad_sum +=if stop_tensor.equal?(node)
          next_op
        else
          _forward_propagate(next_op, node, stop_tensor, nodes_to_compute)
        end
      end

      grad_sum
    end

    def self._compute_derivative(node, grad)
      x = node.items[0] if node.items[0]
      y = node.items[1] if node.items[1]

      case node.operation
      when :add

        sx = tf.shape(x, name: 'add/shape_x')
        sy = tf.shape(y, name: 'add/shape_y')
        rx, ry = _broadcast_gradient_args(sx, sy)

        [tf.reshape(tf.reduce_sum(grad, rx, name: 'add/reduce_sum_x'), sx, name: 'add/reshape_x'),
         tf.reshape(tf.reduce_sum(grad, ry, name: 'add/reduce_sum_y'), sy, name: 'add/reshape_y')]
      when :sub
        sx = tf.shape(x, name: 'sub/shape_x')
        sy = tf.shape(y, name: 'sub/shape_y')
        rx, ry = _broadcast_gradient_args(sx, sy)
        [tf.reshape(tf.reduce_sum(grad, rx), sx), -tf.reshape(tf.reduce_sum(grad, ry), sy)]
      when :mul
        sx = tf.shape(x)
        sy = tf.shape(y)
        rx, ry = _broadcast_gradient_args(sx, sy)

        [ tf.reshape(tf.reduce_sum(tf.mul(grad, y), rx), sx),
          tf.reshape(tf.reduce_sum(tf.mul(x, grad), ry), sy)]
      when :div
        sx = i_op(:shape, x)
        sy = i_op(:shape, y)
        rx, ry = _broadcast_gradient_args(sx, sy)

        [tf.reshape(tf.reduce_sum(tf.div(grad, y), rx), sx),
         tf.reshape(tf.reduce_sum(grad * tf.div(tf.div(-x, y), y),
                                ry), sy)]
      when :matmul
        t_a = node.options[:transpose_a]
        t_b = node.options[:transpose_b]
        grad_a, grad_b = nil
        if !t_a && !t_b
          grad_a = tf.matmul(grad, y, transpose_b: true)
          grad_b = tf.matmul(x, grad, transpose_a: true)
        elsif !ta && tb
          grad_a = tf.matmul(grad, y)
          grad_b = tf.matmul(grad, x, transpose_a: true)
        elsif t_a && !t_b
          grad_a = tf.matmul(y, grad, transpose_b: true)
          grad_b = tf.matmul(x, grad)
        elsif t_a && t_b
          grad_a = tf.matmul(y, grad, transpose_a: true, transpose_b: true)
          grad_b = tf.matmul(grad, x, transpose_a: true, transpose_b: true)
        end

        [grad_a, grad_b]
      when :sin
        grad * tf.cos(x)
      when :tanh
        i_op(:tanh_grad, x, grad)
      when :pow
        z = node
        sx = tf.shape(x)
        sy = tf.shape(y)
        rx, ry = _broadcast_gradient_args(sx, sy)
        gx = tf.reshape(
          tf.reduce_sum(grad * y * tf.pow(x, y - 1), rx), sx)

        log_x = tf.where(x > 0, tf.log(x), tf.zeros_like(x))
        gy = tf.reshape(tf.reduce_sum(grad * z * log_x, ry), sy)

        [gx, gy]
      when :abs
        grad * tf.sign(x)
      when :log
        grad * tf.reciprocal(x)
      when :tanh
        i_op(:tanh_grad, y, grad)
      when :cos
        -grad * tf.sin(x)
      when :max
        x_mask = tf.where(x > y, tf.ones_like(x), tf.zeros_like(y))
        y_mask = tf.where(x < y, tf.zeros_like(x), tf.ones_like(y))
        [x_mask * grad, y_mask * grad]
      when :tan
        secx = tf.reciprocal(tf.cos(x))
        secx2 = tf.square(secx)
        grad * secx2
      when :negate
        -grad
      when :exp
        grad * node
      when :identity
        grad
      when :sum
        _sum_grad(x, y, grad)
      when :reciprocal
        -grad * (tf.constant(1, dtype: x.dtype) / x**2)
      when :sqrt
        tf.constant(1, dtype: x.dtype) / (tf.constant(2, dtype: x.dtype) * tf.sqrt(x)) * grad
      when :stop_gradient
        tf.zeros_like(grad)
      when :square
        y = tf.constant(2.0, dtype: x.dtype)
        tf.multiply(grad, tf.multiply(x, y))
      when :where
        x_mask = i_op(:where, i_op(:ones_like, x), i_op(:zeros_like, y), pred: node.options[:pred])
        y_mask = i_op(:where, i_op(:zeros_like, x), i_op(:ones_like, y), pred: node.options[:pred])
        [x_mask * grad, y_mask * grad]
      when :cond
        x_cond = i_op(:cond, i_op(:ones_like, x), i_op(:zeros_like, y), pred: node.options[:pred])
        y_cond = i_op(:cond, i_op(:zeros_like, x), i_op(:ones_like, x), pred: node.options[:pred])
        [x_cond * grad, y_cond * grad]
      when :mean
        sum_grad = _sum_grad(x, y, grad)
        input_shape = tf.shape(x)
        output_shape = tf.shape(node)
        factor = _safe_shape_div(tf.reduce_prod(input_shape), tf.reduce_prod(output_shape))
        tf.div(sum_grad, tf.cast(factor, sum_grad.data_type))
      else
        raise "no derivative op for #{node}"
      end
    end

    def self._broadcast_gradient_args(input_a, input_b)
      [_op(:broadcast_gradient_args, input_b, input_a), _op(:broadcast_gradient_args, input_a, input_b)]
    end

    def self._broadcast_transform(input_a, input_b)
      _op(:broadcast_transform, input_a, input_b)
    end

    def self._safe_shape_div(x, y)
      x / tf.maximum(y, 1)
    end

    def self._sum_grad(x, y, grad)
      tf.ones_like(x) * grad
    end

    def self._op_supports_broadcast?(node)
      return true if %i[add sub div mul pow].include?(node.operation)
      false
    end

    def self._min_or_max_grad(op, grad)
      y = op
      indicators = tf.cast(tf.equal(y, op.items[0]), grad.data_type)
      num_selected = tf.reduce_sum(indicators, op.items[1])
      _safe_shape_div(indicators, num_selected) * grad
    end

    def self._include?(arr, obj)
      arr.each { |a| return true if a.equal?(obj) }
      false
    end
  end
end