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

      nodes_to_compute = wrt_dx.consumers.select { |t|
        node = tensor.graph.nodes[t]
        node.consumers.include?(tensor.name) || node.equal?(tensor)
      }.compact + [wrt_dx.name]

      grad = i_op(:fill, ts.shape(tensor), ts.constant(1, dtype: wrt_dx.data_type))

      _propagate(grad, tensor, wrt_dx, nodes_to_compute, options[:stop_gradients] || []) || i_op(:zeros_like, wrt_dx)
    end

    def self._propagate(grad, tensor, stop_tensor, nodes_to_compute, stop_gradients = [])
      return grad if stop_tensor.equal?(tensor)
      return nil if stop_gradients && _include?(stop_gradients, tensor)
      return nil unless tensor.is_a?(Operation)

      computed_op = _compute_derivative(tensor, grad)

      if computed_op.is_a?(Array)
        grads = computed_op.each_with_index.collect { |op_grad, index|
          next if op_grad.nil?
          next unless nodes_to_compute.include?(tensor.inputs[index].name)

          _propagate(op_grad, tensor.inputs[index], stop_tensor, nodes_to_compute, stop_gradients)
        }.compact

        return nil if grads.empty?
        grads.size > 1 ? ts.add_n(grads) : grads[0]
      else

        if computed_op.nil?
          return nil
        end
        _propagate(computed_op, tensor.inputs[0], stop_tensor, nodes_to_compute, stop_gradients)
      end
    end

    # TODO: refactor and implement registerGradient
    def self._compute_derivative(node, grad)
      node.graph.name_scope("#{node.name}_grad") do
        x = node.inputs[0] if node.inputs[0]
        y = node.inputs[1] if node.inputs[1]
        z = node.inputs[2] if node.inputs[2]

        case node.operation
        when :add_n
          return [grad] * node.inputs.size
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
        when :squared_difference
          sx = i_op(:shape, x)
          sy = i_op(:shape, y)
          rx, ry = _broadcast_gradient_args(sx, sy)

          x_grad = ts.mul(2.0, grad) * (x - y)

          [ts.reshape(ts.reduce_sum(x_grad, rx), sx),
           ts.reshape(-ts.reduce_sum(x_grad, ry), sy),]
        when :abs
          grad * ts.sign(x)
        when :exp
          grad * node
        when :identity, :print
          grad
        when :tile
          input_shape = ts.shape(x)
          split_shape = ts.reshape(ts.transpose(ts.stack([y, input_shape])), [-1])
          axes = ts.range(0, ts.size(split_shape), 2)
          input_grad = ts.reduce_sum(ts.reshape(grad, split_shape), axes)

          [input_grad, nil]
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
          x_mask = i_op(:where, x, i_op(:ones_like, y), i_op(:zeros_like, z))
          y_mask = i_op(:where, x, i_op(:zeros_like, y), i_op(:ones_like, z))
          [nil, x_mask * grad, y_mask * grad]
        when :mean
          sum_grad = _sum_grad(x, y, grad)[0]
          input_shape = ts.shape(x)
          output_shape = ts.shape(node)
          factor = _safe_shape_div(ts.reduce_prod(input_shape), ts.reduce_prod(output_shape))
          [ts.div(sum_grad, ts.cast(factor, sum_grad.data_type)), nil]
        when :log1p
          grad * ts.reciprocal(i_cons(1, dtype: grad.data_type) + x)
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
        when :zeros_like
          # non differentiable
          nil
        when :transpose
          return [ts.transpose(grad, ts.invert_permutation(y)), nil]
        when :index
          # hack!! not sure how to fix this yet
          return grad if %i[softmax_cross_entropy_with_logits_v2 sparse_softmax_cross_entropy_with_logits].include?(node.inputs[0].operation)

          if node.inputs[0].shape.known? && node.inputs[1].const_value
            multiplier = node.inputs[0].shape.shape[0]
            filler = ts.zeros_like(grad)

            res = Array.new(multiplier) { |index|
              index == node.inputs[1].const_value ? grad : filler
            }
            [res]
          end
        when :squeeze
          _reshape_to_input(node, grad)
        when :concat
          _concat_grad_helper(node, grad, 1, node.inputs.size, 0)
        when :stack
          res = ts.unstack(grad, num: node.inputs.size, axis: node.options[:axis])
          Array.new(node.inputs.size) { |i| res[i] }
        when :unstack
          ts.stack(grad, axis: node.options[:axis])
        when :conv2d
          _Conv2DGrad(node, grad)
        when :flow_dynamic_stitch
          num_values = node.inputs.size / 2
          indices_grad = [nil] * num_values

          inputs = (0...num_values).map { |i| _int32(node, node.inputs[i]) }

          values_grad = inputs.map { |inp| TensorStream.gather(grad, inp) }
          indices_grad + values_grad
        when :gather
          [_op(:gather_grad, grad, node.inputs[1], TensorStream.shape(node.inputs[0])), nil]
        else
          TensorStream::OpMaker.gradient_op(self, node, grad)
        end
      end
    end

    def self._int32(node, x)
      (node.inputs[0].data_type == :int32 ? x : TensorStream.cast(x, :int32))
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

      grad = _op(:case, [_op(:rank, grad).zero?], _op(:tile, new_grad, tile_scaling), _op(:fill, input_shape, grad))

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
      xgrad = ts.where(xmask, grad, zeros, name: "x")
      ygrad = ts.where(xmask, zeros, grad, name: "y")
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

      out_grads = ts.split(grad, sizes, axis: non_neg_concat_dim, num: op.inputs.size - 1)
      end_value_index <= dim_index ? out_grads + [nil] : [nil] + out_grads
    end

    def self._Conv2DGrad(op, grad)
      # dilations = op.get_attr("dilations")
      strides = op.options[:strides]
      padding = op.options[:padding]
      use_cudnn_on_gpu = op.options[:use_cudnn_on_gpu]
      data_format = op.options[:data_format]

      shape_0, shape_1 = ts.shape_n([op.inputs[0], op.inputs[1]])
      [
        _op(:conv2d_backprop_input,
          shape_0,
          op.inputs[1],
          grad,
          strides: strides,
            padding: padding,
            use_cudnn_on_gpu: use_cudnn_on_gpu,
            data_format: data_format),
        _op(:conv2d_backprop_filter,
          op.inputs[0],
          shape_1,
          grad,
          strides: strides,
          padding: padding,
          use_cudnn_on_gpu: use_cudnn_on_gpu,
          data_format: data_format),
      ]
    end
  end
end
