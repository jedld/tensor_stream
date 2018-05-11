module TensorStream
  # Class that provides auto-differentiation
  class MathGradients
    extend TensorStream::OpHelper

    def self.derivative(tensor, dx, options = {})
      gradient_program_name = "_grad_#{tensor.name}_#{dx.name}"
      return options[:graph].get_node(gradient_program_name) if options[:graph] && options[:graph].node_added?(gradient_program_name)

      target_shape = options[:target_shape]
      constant_options = { dtype: options[:dtype] }
      constant_options_1 = { dtype: options[:dtype] || tensor.data_type, shape: target_shape }

      return i_cons(1, constant_options_1) if tensor.equal?(dx)
      return i_cons(0, constant_options) if options[:stop_gradients] && _include?(options[:stop_gradients], tensor)

      if tensor.is_a?(Operation)
        grad = derivative(tensor.items[0], dx, options) if tensor.items[0]
        grad2 = derivative(tensor.items[1], dx, options) if tensor.items[1]

        case tensor.operation
        when :max
          x_mask = i_op(:where, i_op(:ones_like, tensor.items[0]), i_op(:zeros_like, tensor.items[1]), pred: tensor.items[0] > tensor.items[1])
          y_mask = i_op(:where, i_op(:zeros_like, tensor.items[0]), i_op(:ones_like, tensor.items[1]), pred: tensor.items[0] < tensor.items[1])
          x_mask * grad + y_mask * grad2
        when :where
          x_mask = i_op(:where, i_op(:ones_like, tensor.items[0]), i_op(:zeros_like, tensor.items[1]), pred: tensor.options[:pred])
          y_mask = i_op(:where, i_op(:zeros_like, tensor.items[0]), i_op(:ones_like, tensor.items[1]), pred: tensor.options[:pred])
          x_mask * grad + y_mask * grad2
        when :cond
          i_op(:cond, grad, grad2, pred: tensor.options[:pred])
        when :identity, :print, :pad
          grad
        when :negate
          i_cons(-1, constant_options_1) * grad
        when :abs
          grad * i_op(:sign, _ds(tensor.items[0]))
        when :square
          i_cons(2, constant_options_1) * _ds(tensor.items[0]) * grad
        when :exp
          i_op(:exp, tensor.items[0]) * grad
        when :log
          (i_cons(1, constant_options_1) / _ds(tensor.items[0])) * grad
        when :tanh
          (i_cons(1, constant_options_1) - (i_op(:tanh, _ds(tensor.items[0]))**2)) * grad
        when :tan
          (i_cons(1, constant_options_1) / (i_op(:cos, _ds(tensor.items[0]))**2)) * grad
        when :sin
          i_op(:cos, tensor.items[0]) * grad
        when :sqrt
          i_cons(1, constant_options_1) / (i_cons(2, constant_options_1) * i_op(:sqrt, _ds(tensor.items[0]))) * grad
        when :cos
          -i_op(:sin, tensor.items[0]) * grad
        when :add
          grad_with_broadcast(tensor, dx, ->(a,b) { i_op(:add, a, b, name: 'grad_sum') } , options)
        when :sub
          grad_with_broadcast(tensor, dx, ->(a,b) { i_op(:sub, a, b, name: 'grad_sub') } , options)
        when :pow
          gx = _ds(tensor.items[1])*( _ds(tensor.items[0])**(_ds(tensor.items[1]) - 1)) * grad

          log_x = i_op(:where, i_op(:log, tensor.items[0], nil, name: 'log_pow_grad'), i_op(:zeros_like, tensor.items[0]), pred: tensor.items[0] > 0)
          gy = _ds(tensor.items[0])**_ds(tensor.items[1]) * log_x * grad2

          gx + gy
        when :div
          # apply the quotient rule
          gx = i_op(:div, grad, _ds(tensor.items[1]))
          gy = grad2 * i_op(:div, i_op(:div, -_ds(tensor.items[0]), _ds(tensor.items[1])), _ds(tensor.items[1]))

          gx + gy
        when :mul
          # apply the product rule
          grad * _ds(tensor.items[1]) + _ds(tensor.items[0]) * grad2
        when :reduce_mean
          input_size = i_op(:reduce_prod, i_op(:shape, tensor.items[0]))
          output_size = i_op(:reduce_prod, i_op(:shape, tensor))
          factor = input_size / output_size
 
          (grad / i_op(:cast, factor, data_type: grad.dtype))
        when :reduce_sum
          grad
        when :stop_gradient
          return i_cons(0, constant_options)
        when :matmul
          tensor_shape1 = tensor.items[1].shape ? tensor.items[1].shape.shape : nil
          tensor_shape0 = tensor.items[0].shape ? tensor.items[0].shape.shape : nil

          derivative_a = derivative(tensor.items[0], dx, target_shape: target_shape)
          derivative_b = derivative(tensor.items[1], dx, target_shape: target_shape)

          s0 =  i_op(:shape, tensor.items[0])
          s1 =  i_op(:shape, tensor.items[1])

          identity_0 = i_op(:ones, [s0[0], s1[1]], nil, data_type: tensor.items[0].data_type)
          identity_1 = i_op(:ones, [s0[0], s1[1]], nil, data_type: tensor.items[1].data_type)

          matmul_da = i_op(:matmul, identity_0, tensor.items[1], transpose_b: true,
                                                     pad_zeros: true,
                                                     name:        'matrix_dx')
          matmul_db = i_op(:matmul, tensor.items[0], identity_1, transpose_a: true,
                                                     pad_zeros: true,
                                                     name:        'matrix_dy')       

          zero_vect = i_op(:zeros, target_shape, nil, name: 'zero_vect')

          # matmul_db = op(:transpose, matmul_db, nil).first

          # begin_a = op(:zeros, op(:rank, matmul_db), nil, data_type: :int32, name: 'begin_a')
          # matmul_b_shape = op(:shape, matmul_db)
          # end_a = [matmul_b_shape[0], 1]

          matmul_da = i_op(:cond, matmul_da[0], matmul_da, pred: op(:rank, derivative_a) > 0)

          # matmul_da = op(:cond, matmul_da[0], matmul_da, pred: op(:rank, derivative_a) > 0)
          norm_a = i_op(:mul, derivative_a, matmul_da, name: 'grad_a_norm_mul_da')
          norm_b = i_op(:mul, derivative_b, matmul_db, name: 'grad_b_norm_mul_db')

          # norm_a = i_op(:cond, norm_a[0], norm_a, pred: i_op(:rank, matmul_da) > i_op(:rank, derivative_a))
          # norm_b = i_op(:cond, norm_b[0], norm_b, pred: i_op(:rank, matmul_db) > i_op(:rank, derivative_b))

          i_op(:cond, norm_a, zero_vect, pred: i_op(:reduce_sum, norm_a) != 0) + i_op(:cond, norm_b, zero_vect, pred: i_op(:reduce_sum, norm_b) != 0)
        else
          raise "no derivative implementation found for op #{tensor.operation}"
        end
      elsif tensor.is_a?(TensorStream::Variable)
        i_cons(0, constant_options)
      elsif tensor.is_a?(TensorStream::Placeholder)
        i_cons(0, constant_options)
      else
        i_cons(0, constant_options)
      end.tap do |ops|
        options[:graph].add_node!(gradient_program_name, ops) if options[:graph]
      end
    end

    def self._ds(tensor)
      return tensor unless tensor.is_a?(Operation)

      case tensor.operation
      when :reduce_sum
        tensor.items[0]
      else
        tensor
      end
    end

    def self.grad_with_broadcast(tensor, dx, func, options)
      grad = derivative(tensor.items[0], dx, options)
      grad2 = derivative(tensor.items[1], dx, options)
      elements1 = i_op(:reduce_prod, i_op(:shape, tensor.items[0]), data_type: :float32)
      elements2 = i_op(:reduce_prod, i_op(:shape, tensor.items[1]), data_type: :float32)
      multiplier = elements1 / elements2
      func.call(grad, grad2 * multiplier)
    end

    def self._include?(arr, obj)
      arr.each { |a| return true if a.equal?(obj) }
      false
    end
  end
end