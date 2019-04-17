module TensorStream
  module MathOps
    def self.included(klass)
      klass.class_eval do
        register_op :tanh, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| Math.tanh(t) }
        end

        register_op :tan, no_eval: true do |context, tensor, inputs|
          call_op(inputs[0], context) { |t, _b| Math.tan(t) }
        end

        register_op :atan, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| Math.atan(t) }
        end

        register_op :sin, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| Math.sin(t) }
        end

        register_op :add, no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :add, a, b, context) { |t, u| t + u }
        end

        register_op :add_n, no_eval: true do |context, tensor, inputs|
          if inputs.size == 1
            complete_eval(inputs[0], context)
          elsif inputs.size > 1

            a = inputs.pop
            until inputs.empty?
              b = inputs.pop
              a = call_vector_op(tensor, :add, a, b, context) { |t, u| t + u }
            end
            a
          end
        end

        register_op :bias_add do |_context, _tensor, inputs|
          value, bias = inputs
          arr = value.flatten.each_slice(bias.size).map do |slice|
            slice.each_with_index.map { |elem, index| elem + bias[index] }
          end
          TensorShape.reshape(arr, shape_eval(value))
        end

        register_op :bias_add_grad do |_context, _tensor, inputs|
          received_grad = inputs[0]
          bias_size = shape_eval(received_grad).last
          grad_sum = Array.new(bias_size) { 0.0 }
          received_grad.flatten.each_slice(bias_size) do |slice|
            slice.each_with_index.map { |elem, index| grad_sum[index] += elem }
          end
          grad_sum
        end

        register_op :sub, no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :sub, a, b, context) { |t, u| t - u }
        end

        register_op %i[floor_mod mod], no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :mod, a, b, context) { |t, u| t % u }
        end

        register_op %i[floor_div], no_eval: true do |context, tensor, inputs|
          a, b = inputs
          if fp_type?(tensor.data_type)
            call_vector_op(tensor, :div, a, b, context) { |t, u| (t / u).to_i.to_f }
          else
            call_vector_op(tensor, :div, a, b, context) { |t, u| t / u }
          end
        end

        register_op :mul, no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :mul, a, b, context) { |t, u| t * u }
        end

        register_op :pow, no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :pow, a, b, context) { |t, u| t**u }
        end

        register_op :squared_difference, no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :squared_difference, a, b, context) { |t, u| (t - u) * (t - u) }
        end

        register_op :round, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b|  t.round }
        end

        register_op :abs, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| t.abs }
        end

        register_op :asin, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| Math.asin(t) }
        end

        register_op :acos, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| Math.acos(t) }
        end

        register_op :cos, no_eval: true do |context, tensor, inputs|
          call_op(inputs[0], context) { |t, _b| Math.cos(t) }
        end

        register_op :log1p, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| Math.log(1 + t) }
        end

        register_op :log, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| t < 0 ? Float::NAN : Math.log(t) }
        end

        register_op :exp, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| Math.exp(t) }
        end

        register_op :sigmoid, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| sigmoid(t) }
        end

        register_op :sqrt, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| Math.sqrt(t) }
        end

        register_op :rsqrt, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b|  1 / Math.sqrt(t) }
        end

        register_op :rsqrt_grad, no_eval: true do |context, tensor, inputs|
          y, grad = inputs
          call_vector_op(tensor, :rsqrt_grad, y, grad, context) { |_y, g| 0.5 * g * (_y ** 3) }
        end

        register_op :floor, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| t.floor }
        end

        register_op :ceil, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| t.ceil }
        end

        register_op :square, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| t * t }
        end

        register_op :reciprocal, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| 1 / t }
        end

        register_op %i[neg negate], no_eval: true do |context, tensor, inputs|
          call_vector_op(tensor, :negate, inputs[0], nil, context) { |t, _u| -t }
        end

        register_op :tanh_grad, no_eval: true do |context, _tensor, inputs|
          call_op(inputs[0], context) { |t, _b| 1 - Math.tanh(t) * Math.tanh(t) }
        end

        register_op :top_k do |context, tensor, inputs|
          values, k = inputs
          v_shape = shape_eval(values)

          sorted = tensor.options[:sorted]
          work_values = TensorShape.reshape(values, [-1, v_shape.last])
          work_values.map! do |row|
            last_k = row.map.with_index { |r, index| [r, index] }.sort! { |a,b| a[0] <=> b[0] }.last(k)
            last_k.reverse! if sorted
            last_k
          end

          top_k = work_values.map { |row| row.map { |r| r[0] } }
          top_indices = work_values.map { |row| row.map { |r| r[1] } }
          v_shape[-1] = k

          TensorStream::Evaluator::OutputGroup.new([TensorShape.reshape(top_k, v_shape), TensorShape.reshape(top_indices, v_shape)], [tensor.inputs[0].data_type, :int32])
        end

        register_op(%i[argmax arg_max]) do |_context, tensor, inputs|
          axis = inputs[1] || 0
          rank = get_rank(inputs[0])
          raise TensorStream::InvalidArgumentError, "Expected dimension in the range [#{-rank},#{rank}) but got #{axis}" if axis < -rank || axis >= rank

          new_shape = shape_eval(inputs[0])
          ns = new_shape.each_with_index.collect { |shape, index|
            next nil if index == axis

            shape
          }.compact

          Tensor.cast_dtype(TensorShape.reshape(get_op_with_axis(inputs[0], axis, 0, :max), ns), tensor.options[:output_type])
        end

        register_op(%i[argmin arg_min]) do |_context, tensor, inputs|
          axis = inputs[1] || 0
          rank = get_rank(inputs[0])
          raise TensorStream::InvalidArgumentError, "Expected dimension in the range [#{-rank},#{rank}) but got #{axis}" if axis < -rank || axis >= rank

          new_shape = shape_eval(inputs[0])
          ns = new_shape.each_with_index.collect { |shape, index|
            next nil if index == axis

            shape
          }.compact

          Tensor.cast_dtype(TensorShape.reshape(get_op_with_axis(inputs[0], axis, 0, :min), ns), tensor.options[:output_type])
        end

        register_op :cumprod do |context, tensor, inputs|
          c = fp_type?(tensor.data_type) ? 1.0 : 1
          reverse_option = tensor.options[:reverse]
          exclusive = tensor.options[:exclusive]

          reduction(context, tensor) do |arr|
            if arr.nil?
              c
            else
              count = arr.size
              arr = arr.reverse if reverse_option
              arr = [1] + arr if exclusive

              start_prod = arr[0]
              mapped = arr[1...count].map { |v|
                start_prod = vector_op(start_prod, v) { |a, b| a * b }
              }

              arr = [arr[0]] + mapped
              reverse_option ? arr.reverse : arr
            end
          end
        end

        register_op :sum, noop: true do |context, tensor, _inputs|
          reduction(context, tensor) do |arr|
            reduced_val = arr[0]
            arr[1..arr.size].each do |v|
              reduced_val = vector_op(reduced_val, v) { |t, u| t + u }
            end
            reduced_val
          end
        end

        register_op :prod, noop: true do |context, tensor, _inputs|
          c = fp_type?(tensor.data_type) ? 1.0 : 1
          reduction(context, tensor) do |arr|
            if arr.nil?
              c
            else
              reduced_val = arr[0]
              arr[1..arr.size].each do |v|
                reduced_val = vector_op(reduced_val, v) { |a, b| a * b }
              end
              reduced_val
            end
          end
        end

        register_op :sigmoid_grad, no_eval: true do |context, tensor, inputs|
          a, b = inputs
          call_vector_op(tensor, :sigmoid_grad, a, b, context) { |t, u| u * sigmoid(t) * (1 - sigmoid(t)) }
        end

        register_op :mean, noop: true do |context, tensor, _inputs|
          c = fp_type?(tensor.data_type) ? 0.0 : 0

          reduction(context, tensor) do |arr|
            return c if arr.nil?

            reduced_val = arr[0]
            arr[1..arr.size].each do |v|
              reduced_val = vector_op(reduced_val, v) { |a, b| a + b }
            end

            vector_op(reduced_val, nil) { |a, _b| a / arr.size }
          end
        end

        register_op :mat_mul do |_context, tensor, inputs|
          matrix_a, matrix_b = inputs
          rank_a = get_rank(matrix_a)
          rank_b = get_rank(matrix_b)
          raise "#{tensor.inputs[0].name} rank must be greater than 1" if rank_a < 2
          raise "#{tensor.inputs[1].name} rank must be greater than 1" if rank_b < 2

          # check matrix dimensions
          if rank_a >= 3
            matrix_a.zip(matrix_b).map do |m_a, m_b|
              matmul(m_a, m_b, tensor)
            end
          else
            matmul(matrix_a, matrix_b, tensor)
          end
        end

        def matmul(m_a, m_b, tensor)
          m_a = m_a.transpose if tensor.options[:transpose_a]
          m_b = m_b.transpose if tensor.options[:transpose_b]
          raise TensorStream::ValueError, "incompatible shape sizes for matrix multiplication (#{m_a[0].size} != #{m_b.size}) #{shape_eval(m_a)} vs #{shape_eval(m_b)}" if m_a[0].size != m_b.size

          (Matrix[*m_a] * Matrix[*m_b]).to_a
        end

        register_op %i[max maximum], noop: true do |context, tensor, inputs|
          call_vector_op(tensor, :max, inputs[0], inputs[1], context) { |t, u| [t, u].max }
        end

        register_op %i[min minimum], noop: true do |context, tensor, inputs|
          call_vector_op(tensor, :min, inputs[0], inputs[1], context) { |t, u| [t, u].min }
        end

        def reduction(child_context, tensor, &block)
          val = global_eval(tensor, tensor.inputs[0], child_context)
          axis = global_eval(tensor, tensor.inputs[1], child_context)
          keep_dims = global_eval(tensor, tensor.options[:keepdims], child_context)

          reduce(val, axis, keep_dims, &block)
        end
      end
    end
  end
end
