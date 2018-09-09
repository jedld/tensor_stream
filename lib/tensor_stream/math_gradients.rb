module TensorStream
  # Class that provides auto-differentiation
  # Most gradients are ported over from tensorflow's math_grad.py
  class MathGradients
    extend TensorStream::OpHelper

    def self.ts
      TensorStream
    end

    def self.derivative(tensor, wrt_dx, options = {})
      return i_op(:ones_like, tensor) if tensor.equal?(wrt_dx)
      return i_op(:zeros_like, wrt_dx) unless wrt_dx.consumers.include?(tensor.name)

      nodes_to_compute = wrt_dx.consumers.select do |t|
        node = tensor.graph.nodes[t]
        node.consumers.include?(tensor.name) || node.equal?(tensor)
      end.compact + [wrt_dx.name]

      grad = i_op(:fill, ts.shape(tensor), ts.constant(1, dtype: wrt_dx.data_type))

      _propagate(grad, tensor, wrt_dx, nodes_to_compute, options[:stop_gradients] || []) || i_op(:zeros_like, wrt_dx)
    end

    def self._propagate(grad, tensor, stop_tensor, nodes_to_compute, stop_gradients = [])
      return grad if stop_tensor.equal?(tensor)
      return nil if stop_gradients && _include?(stop_gradients, tensor)
      return nil unless tensor.is_a?(Operation)

      computed_op = _compute_derivative(tensor, grad)

      if computed_op.is_a?(Array)
        computed_op.each_with_index.collect do |op_grad, index|
          next if op_grad.nil?
          next unless nodes_to_compute.include?(tensor.inputs[index].name)

          _propagate(op_grad, tensor.inputs[index], stop_tensor, nodes_to_compute, stop_gradients)
        end.compact.reduce(:+)
      else
        return nil if computed_op.nil?
        _propagate(computed_op, tensor.inputs[0], stop_tensor, nodes_to_compute, stop_gradients)
      end
    end

    #TODO: refactor and implement registerGradient
    def self._compute_derivative(node, grad)
      node.graph.name_scope("#{node.name}_grad") do
        x = node.inputs[0] if node.inputs[0]
        y = node.inputs[1] if node.inputs[1]

        case node.operation
        when :add_n
          return [grad] * node.inputs.size
        when :add
          return [grad, grad] if shapes_fully_specified_and_equal(x, y)
          sx = ts.shape(x, name: 'add/shape_x')
          sy = ts.shape(y, name: 'add/shape_y')
          rx, ry = _broadcast_gradient_args(sx, sy)

          [ts.reshape(ts.reduce_sum(grad, rx, name: 'add/reduce_sum_x'), sx),
           ts.reshape(ts.reduce_sum(grad, ry, name: 'add/reduce_sum_y'), sy)]
        when :asin
          ts.control_dependencies([grad]) do
            x2 = ts.square(x)
            one = ts.constant(1, dtype: grad.data_type)
            den = ts.sqrt(ts.subtract(one, x2))
            inv = ts.reciprocal(den)
            grad * inv
          end
        when :acos
          ts.control_dependencies([grad]) do
            x2 = ts.square(x)
            one = ts.constant(1, dtype: grad.data_type)
            den = ts.sqrt(ts.subtract(one, x2))
            inv = ts.reciprocal(den)
            -grad * inv
          end
        when :atan
          ts.control_dependencies([grad]) do
            x2 = ts.square(x)
            one = ts.constant(1, dtype: grad.data_type)
            inv = ts.reciprocal(ts.add(one, x2))
            grad * inv
          end
        when :fill
          [nil, ts.reduce_sum(grad)]
        when :sub
          return [grad, -grad] if shapes_fully_specified_and_equal(x, y)

          sx = ts.shape(x, name: 'sub/shape_x')
          sy = ts.shape(y, name: 'sub/shape_y')
          rx, ry = _broadcast_gradient_args(sx, sy)

          [ts.reshape(ts.reduce_sum(grad, rx, name: 'add/reduce_sub_x'), sx),
           -ts.reshape(ts.reduce_sum(grad, ry, name: 'add/reduce_sub_y'), sy)]
        when :mul
          sx = ts.shape(x)
          sy = ts.shape(y)
          rx, ry = _broadcast_gradient_args(sx, sy)

          [ts.reshape(ts.reduce_sum(ts.mul(grad, y), rx), sx),
           ts.reshape(ts.reduce_sum(ts.mul(x, grad), ry), sy)]
        when :div
          sx = i_op(:shape, x)
          sy = i_op(:shape, y)
          rx, ry = _broadcast_gradient_args(sx, sy)

          [ts.reshape(ts.reduce_sum(ts.div(grad, y), rx), sx),
           ts.reshape(ts.reduce_sum(grad * ts.div(ts.div(-x, y), y), ry), sy)]
        when :mod
          sx = ts.shape(x)
          sy = ts.shape(y)
          rx, ry = _broadcast_gradient_args(sx, sy)
          floor_xy = ts.floor_div(x, y)
          gx = ts.reshape(ts.reduce_sum(grad, rx), sx)
          gy = ts.reshape(ts.reduce_sum(grad * ts.negative(floor_xy), ry), sy)

          [gx, gy]
        when :prod
          input_shape = ts.shape(x)
          y = ts.range(0, ts.rank(x)) if y.nil?
          reduction_indices = ts.reshape(y, [-1])

          output_shape_kept_dims = ts.reduced_shape(input_shape, y)
          tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims)
          grad = ts.reshape(grad, output_shape_kept_dims)
          grad = ts.tile(grad, tile_scaling)

          perm, reduced_num, other_num = ts.device("/cpu:0") do
            rank = ts.rank(x)
            reduction_indices = (reduction_indices + rank) % rank
            reduced = ts.cast(reduction_indices, :int32)
            idx = ts.range(0, rank)
            other, = ts.setdiff1d(idx, reduced)

            [ts.concat([reduced, other], 0),
             ts.reduce_prod(ts.gather(input_shape, reduced)),
             ts.reduce_prod(ts.gather(input_shape, other))]
          end

          permuted = ts.transpose(x, perm)
          permuted_shape = ts.shape(permuted)

          reshaped = ts.reshape(permuted, [reduced_num, other_num])

          # Calculate product, leaving out the current entry
          left = ts.cumprod(reshaped, axis: 0, exclusive: true)
          right = ts.cumprod(reshaped, axis: 0, exclusive: true, reverse: true)
          y = ts.reshape(left * right, permuted_shape)

          # Invert the transpose and reshape operations.
          # Make sure to set the statically known shape information through a reshape.
          out = grad * ts.transpose(y, ts.invert_permutation(perm))
          [ts.reshape(out, input_shape, name: 'prod'), nil]
        when :squared_difference
          sx = i_op(:shape, x)
          sy = i_op(:shape, y)
          rx, ry = _broadcast_gradient_args(sx, sy)

          x_grad = ts.mul(2.0, grad) * (x - y)

          [ts.reshape(ts.reduce_sum(x_grad, rx), sx),
           ts.reshape(-ts.reduce_sum(x_grad, ry), sy)]
        when :mat_mul
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
        when :sin
          grad * ts.cos(x)
        when :tanh
          grad * i_op(:tanh_grad, x)
        when :pow
          z = node
          sx = ts.shape(x)
          sy = ts.shape(y)
          rx, ry = _broadcast_gradient_args(sx, sy)
          gx = ts.reduce_sum(grad * y * ts.pow(x, y - 1), rx)

          log_x = ts.where(x > 0, ts.log(x), ts.zeros_like(x))
          gy = ts.reduce_sum(grad * z * log_x, ry)

          [gx, gy]
        when :abs
          grad * ts.sign(x)
        when :log
          grad * ts.reciprocal(x)
        when :cos
          -grad * ts.sin(x)
        when :max
          _min_or_max_grad(node.inputs, grad, ->(a, b) { ts.greater_equal(a, b) })
        when :min
          _min_or_max_grad(node.inputs, grad, ->(a, b) { ts.less_equal(a, b) })
        when :tan
          secx = ts.reciprocal(ts.cos(x))
          secx2 = ts.square(secx)
          grad * secx2
        when :negate
          -grad
        when :exp
          grad * node
        when :identity, :print
          grad
        when :sign
          ts.zeros(ts.shape(x), dtype: x.data_type)
        when :tile
          input_shape = ts.shape(x)
          split_shape = ts.reshape(ts.transpose(ts.stack([y, input_shape])), [-1])
          axes = ts.range(0, ts.size(split_shape), 2)
          input_grad = ts.reduce_sum(ts.reshape(grad, split_shape), axes)

          [input_grad, nil]
        when :sum
          _sum_grad(x, y, grad)
        when :reciprocal
          -grad * (ts.constant(1, dtype: x.dtype) / x**2)
        when :sqrt
          ts.constant(1, dtype: x.dtype) / (ts.constant(2, dtype: x.dtype) * ts.sqrt(x)) * grad
        when :stop_gradient
          ts.zeros_like(grad)
        when :square
          y = ts.constant(2.0, dtype: x.dtype)
          ts.multiply(grad, ts.multiply(x, y))
        when :where
          x_mask = i_op(:where, i_op(:ones_like, x), i_op(:zeros_like, y), pred: node.options[:pred])
          y_mask = i_op(:where, i_op(:zeros_like, x), i_op(:ones_like, y), pred: node.options[:pred])
          [x_mask * grad, y_mask * grad]
        when :cond
          x_cond = i_op(:cond, i_op(:ones_like, x), i_op(:zeros_like, y), pred: node.options[:pred])
          y_cond = i_op(:cond, i_op(:zeros_like, x), i_op(:ones_like, x), pred: node.options[:pred])
          [x_cond * grad, y_cond * grad]
        when :mean
          sum_grad = _sum_grad(x, y, grad)[0]
          input_shape = ts.shape(x)
          output_shape = ts.shape(node)
          factor = _safe_shape_div(ts.reduce_prod(input_shape), ts.reduce_prod(output_shape))
          [ts.div(sum_grad, ts.cast(factor, sum_grad.data_type)), nil]
        when :log1p
          grad * ts.reciprocal(i_cons(1, dtype: grad.data_type) + x)
        when :sigmoid
          i_op(:sigmoid_grad, x, grad)
        when :sigmoid_grad
          gb = grad * y
          [gb - 2.0 * gb * x, i_op(:sigmoid_grad, x, grad)]
        when :softmax
          i_op(:softmax_grad, x, grad)
        when :softmax_cross_entropy_with_logits_v2
          output = node
          logits = node.inputs[0]
          [_broadcast_mul(grad, output[1]), -ts.nn.log_softmax(logits)]
        when :sparse_softmax_cross_entropy_with_logits
          output = node
           [_broadcast_mul(grad, output[1]), nil]
        when :floor, :ceil
          # non differentiable
          nil
        when :zeros_like
          # non differentiable
          nil
        when :argmin, :argmax, :floor_div
          # non differentiable
          [nil, nil]
        when :transpose
          return [ts.transpose(grad, ts.invert_permutation(y)), nil]
        when :index
          #hack!! not sure how to fix this yet
          return grad if %i[softmax_cross_entropy_with_logits_v2 sparse_softmax_cross_entropy_with_logits].include?(node.inputs[0].operation)

          if node.inputs[0].shape.known? && node.inputs[1].value
            multiplier = node.inputs[0].shape.shape[0]
            filler = ts.zeros_like(grad)

            res = Array.new(multiplier) { |index|
              index == node.inputs[1].value ? grad : filler
            }
            [res]
          end
        when :squeeze
          _reshape_to_input(node, grad)
        when :expand_dims
          [_reshape_to_input(node, grad), nil]
        when :concat
          _concat_grad_helper(node, grad, 1, node.inputs.size, 0)
        when :reshape
          [ts.reshape(grad, ts.shape(node.inputs[0])), nil]
        when :stack
          res = ts.unstack(grad, num: node.inputs.size, axis: node.options[:axis])
          Array.new(node.inputs.size) { |i| res[i] }
        when :unstack
          ts.stack(grad, axis: node.options[:axis])
        when :cast
          t = %i[float16 float32 float64]
          src_type = node.inputs[0].data_type
          dst_type = grad.data_type

          if t.key?(src_type) && t.key?(dst_type)
            ts.cast(grad, src_type)
          else
            nil
          end
        else
          raise "no derivative op for #{node.operation}"
        end
      end
    end

    def self._reshape_to_input(node, grad)
      ts.reshape(grad, ts.shape(node.inputs[0]))
    end

    def self._broadcast_gradient_args(input_a, input_b)
      res = _op(:broadcast_gradient_args, input_a, input_b)
      [res[0], res[1]]
    end

    def self._broadcast_transform(input_a, input_b)
      _op(:broadcast_transform, input_a, input_b)
    end

    def self._safe_shape_div(arg_x, arg_y)
      _op(:floor_div, arg_x, ts.maximum(arg_y, 1))
    end

    def self._sum_grad(arg_x, arg_y, grad)
      input_shape = _op(:shape, arg_x)
      output_shape_kept_dims = ts.reduced_shape(input_shape, arg_y)
      tile_scaling = _safe_shape_div(input_shape, output_shape_kept_dims)
      new_grad = _op(:reshape, grad, output_shape_kept_dims)

      grad = _op(:cond, _op(:fill, input_shape, grad), _op(:tile, new_grad, tile_scaling), pred: _op(:rank, grad).zero?)

      [grad, nil]
    end

    def self._op_supports_broadcast?(node)
      return true if %i[add sub div mul pow].include?(node.operation)
      false
    end

    def self._min_or_max_grad(inputs, grad, selector_op)
      x = inputs[0]
      y = inputs[1]
      gdtype = grad.data_type
      sx = ts.shape(x)
      sy = ts.shape(y)
      gradshape = ts.shape(grad)
      zeros = ts.zeros(gradshape, dtype: gdtype)
      xmask = selector_op.call(x, y)
      rx, ry = _broadcast_gradient_args(sx, sy)
      xgrad = ts.where(xmask, grad, zeros, name: 'x')
      ygrad = ts.where(xmask, zeros, grad, name: 'y')
      gx = ts.reshape(ts.reduce_sum(xgrad, rx), sx)
      gy = ts.reshape(ts.reduce_sum(ygrad, ry), sy)
      [gx, gy]
    end

    def self._broadcast_mul(vec, mat)
      vec = ts.expand_dims(vec, -1)
      vec * mat
    end

    def self._include?(arr, obj)
      arr.each { |a| return true if a.equal?(obj) }
      false
    end

    def self._extract_input_shapes(inputs)
      sizes = []
      fully_known = true
      inputs.each do |x|
        input_shape = ts.shape(x)
        unless input_shape.is_const
          fully_known = false
          break
        end
        sizes << input_shape.value
      end

      if fully_known
        sizes
      else
        ts.shape_n(inputs)
      end
    end

    def self._concat_grad_helper(op, grad, start_value_index, end_value_index, dim_index)
      # Degenerate concatenation, just return grad.
      if op.inputs.size == 2
        return end_value_index <= dim_index ? [grad] + [nil] : [nil] + [grad]
      end
      concat_dim = op.inputs[dim_index]
      input_values = op.inputs[start_value_index..end_value_index]
      non_neg_concat_dim = concat_dim % ts.rank(input_values[0])
      sizes = _extract_input_shapes(input_values)

      slicer = ts.slice(ts.stack(sizes, axis: 1), [non_neg_concat_dim, 0], [1, -1])
      sizes = ts.squeeze(slicer)

      out_grads = ts.split(grad, sizes, axis: non_neg_concat_dim, num: input_values.size)
      end_value_index <= dim_index ? out_grads + [nil] : [nil] + out_grads
    end
  end
end
