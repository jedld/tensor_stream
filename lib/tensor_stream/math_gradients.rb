module TensorStream
  # Class that provides auto-differentiation
  class MathGradients
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

      grad = i_op(:fill, tf.shape(tensor), tf.constant(1, dtype: wrt_dx.data_type))

      _propagate(grad, tensor, wrt_dx, nodes_to_compute, options[:stop_gradients] || [] ) || i_op(:zeros_like, wrt_dx)
    end

    def self._propagate(grad, tensor, stop_tensor, nodes_to_compute, stop_gradients = [])
      return grad if stop_tensor.equal?(tensor)
      return nil if stop_gradients && _include?(stop_gradients, tensor)
      return nil unless tensor.is_a?(Operation)

      computed_op = _compute_derivative(tensor, grad)

      if computed_op.is_a?(Array)
        partials = []
        computed_op.each_with_index do |op_grad, index|
          next if op_grad.nil?

          if nodes_to_compute.include?(tensor.inputs[index].name)
            partials << _propagate(op_grad, tensor.inputs[index], stop_tensor, nodes_to_compute, stop_gradients)
          end
        end

        partials.compact.reduce(:+)
      else
        return nil if computed_op.nil?
        _propagate(computed_op, tensor.inputs[0], stop_tensor, nodes_to_compute, stop_gradients)
      end
    end

    def self._compute_derivative(node, grad)
      node.graph.name_scope("#{node.name}_grad") do
        x = node.inputs[0] if node.inputs[0]
        y = node.inputs[1] if node.inputs[1]

        case node.operation
        when :add
          return [grad, grad] if shapes_fully_specified_and_equal(x, y)
          sx = tf.shape(x, name: 'add/shape_x')
          sy = tf.shape(y, name: 'add/shape_y')
          rx, ry = _broadcast_gradient_args(sx, sy)

          [ tf.reshape(tf.reduce_sum(grad, rx, name: 'add/reduce_sum_x'),sx),
            tf.reshape(tf.reduce_sum(grad, ry, name: 'add/reduce_sum_y'),sy) ]
        when :sub
          return [grad, -grad] if shapes_fully_specified_and_equal(x, y)

          sx = tf.shape(x, name: 'sub/shape_x')
          sy = tf.shape(y, name: 'sub/shape_y')
          rx, ry = _broadcast_gradient_args(sx, sy)

          [ tf.reshape(tf.reduce_sum(grad, rx, name: 'add/reduce_sub_x'),sx),
            -tf.reshape(tf.reduce_sum(grad, ry, name: 'add/reduce_sub_y'),sy) ]
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

          [
            tf.reshape(tf.reduce_sum(tf.div(grad, y), rx), sx),
            tf.reshape(tf.reduce_sum(grad * tf.div(tf.div(-x, y), y),
                                  ry), sy)]
        when :mod
          sx = tf.shape(x)
          sy = tf.shape(y)
          rx, ry = _broadcast_gradient_args(sx, sy)
          floor_xy = tf.floor_div(x, y)
          gx = tf.reshape(tf.reduce_sum(grad, rx), sx)
          gy = tf.reshape(tf.reduce_sum(grad * tf.negative(floor_xy), ry), sy)

          [gx, gy]
        when :squared_difference
          sx = i_op(:shape, x)
          sy = i_op(:shape, y)
          rx, ry = _broadcast_gradient_args(sx, sy)

          x_grad = tf.mul(2.0, grad) * (x - y)

          [ tf.reshape(tf.reduce_sum(x_grad, rx), sx),
            tf.reshape(-tf.reduce_sum(x_grad, ry), sy)]
        when :matmul
          t_a = node.options[:transpose_a]
          t_b = node.options[:transpose_b]

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
          grad * i_op(:tanh_grad, x)
        when :pow
          z = node
          sx = tf.shape(x)
          sy = tf.shape(y)
          rx, ry = _broadcast_gradient_args(sx, sy)
          gx = tf.reduce_sum(grad * y * tf.pow(x, y - 1), rx)

          log_x = tf.where(x > 0, tf.log(x), tf.zeros_like(x))
          gy = tf.reduce_sum(grad * z * log_x, ry)

          [gx, gy]
        when :abs
          grad * tf.sign(x)
        when :log
          grad * tf.reciprocal(x)
        when :tanh
          i_op(:tanh_grad, x) * grad
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
        when :identity, :print
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
          sum_grad  = _sum_grad(x, y, grad)[0]
          input_shape = tf.shape(x)
          output_shape = tf.shape(node)
          factor = _safe_shape_div(tf.reduce_prod(input_shape), tf.reduce_prod(output_shape))
          tf.div(sum_grad, tf.cast(factor, sum_grad.data_type))
        when :log1p
          grad * tf.reciprocal(i_cons(1, dtype: grad.data_type) + x)
        when :sigmoid
          i_op(:sigmoid_grad, x, grad)
        when :softmax
          i_op(:softmax_grad, x, grad)
        when :softmax_cross_entropy_with_logits_v2
          # -grad * tf.reciprocal(i_op(:softmax, x))
          [i_op(:softmax_cross_entropy_with_logits_v2_grad, x, y, grad), nil]
          # i_op(:softmax_grad, x, -grad * tf.reciprocal(i_op(:softmax, x)))
        when :floor, :ceil
          # non differentiable
          nil
        when :zeros_like
          # non differentiable
          nil
        when :argmin, :argmax
          # non differentiable
          [nil, nil]
        else
          raise "no derivative op for #{node.operation}"
        end
      end
    end

    def self._broadcast_gradient_args(input_a, input_b)
      res = _op(:broadcast_gradient_args, input_a, input_b)
      [res[0], res[1]]
    end

    def self._broadcast_transform(input_a, input_b)
      _op(:broadcast_transform, input_a, input_b)
    end

    def self._safe_shape_div(x, y)
      _op(:floor_div, x , tf.maximum(y, 1))
    end

    def self._sum_grad(x, y, grad)
      input_shape = _op(:shape, x)
      output_shape_kept_dims = tf.reduced_shape(input_shape, y)
      tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims)
      new_grad = _op(:reshape, grad, output_shape_kept_dims)

      grad = _op(:cond, _op(:fill, input_shape, grad) , _op(:tile, new_grad, tile_scaling), pred: _op(:rank, grad) == 0 )

      [grad, nil ]
    end

    def self._op_supports_broadcast?(node)
      return true if %i[add sub div mul pow].include?(node.operation)
      false
    end

    def self._min_or_max_grad(op, grad)
      y = op
      indicators = tf.cast(tf.equal(y, op.inputs[0]), grad.data_type)
      num_selected = tf.reduce_sum(indicators, op.inputs[1])
      _safe_shape_div(indicators, num_selected) * grad
    end

    def self._include?(arr, obj)
      arr.each { |a| return true if a.equal?(obj) }
      false
    end
  end
end