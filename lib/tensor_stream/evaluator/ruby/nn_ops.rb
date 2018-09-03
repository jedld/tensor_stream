module TensorStream
  ## Collection of machine learning related ops
  module NNOps
    def NNOps.included(klass)
      klass.class_eval do
        register_op :apply_gradient_descent do |context, tensor, inputs|
          target_var, learning_rate, delta = inputs
          assign = tensor.inputs[0] || tensor

          assign.value = process_vector_math_op(tensor, target_var, delta, context, ->(t, u) { t - u * learning_rate })
          assign.value
        end

        register_op :apply_momentum do |context, tensor, inputs|
          target_var, momentum_var, learning_rate, grad, momentum = inputs
          assign = tensor.inputs[0] || tensor
          assign_acc = tensor.inputs[1]
          assign_acc.value = process_vector_math_op(tensor, momentum_var, grad, context, ->(t, u) { t * momentum + u })
          if tensor.options[:use_nesterov]
            delta = process_vector_math_op(tensor, grad, momentum_var, context, ->(g, acc) { g * learning_rate + acc * momentum * learning_rate })
            assign.value = process_vector_math_op(tensor,target_var, delta, context, ->(t, u) { t - u })
          else
            assign.value = process_vector_math_op(tensor, target_var, momentum_var, context, ->(v, acc) { v - acc * learning_rate })
          end
          assign.value
        end

        register_op :apply_adam do |context, tensor, inputs|
          target_var, m, v, beta1_power, beta2_power, lr_t, beta1_t, beta2_t, epsilon_t, grad = inputs
          alpha = lr_t * Math.sqrt( 1.0 - beta2_power) / (1.0 - beta1_power)
          assign = tensor.inputs[0]
          assign_m = tensor.inputs[1]
          assign_v = tensor.inputs[2]

          m_delta = process_vector_math_op(tensor, grad, m, context, ->(g, m_d) { (g - m_d) * (1.0 - beta1_t) })
          assign_m.value = process_vector_math_op(tensor, m, m_delta, context, ->(u_d , v_d) { u_d + v_d })
          assign_v.value = process_vector_math_op(tensor, v, grad, context, ->(u_d , v_d) { u_d + (v_d ** 2 - u_d) * (1.0 - beta2_t)})
          v_delta = process_vector_math_op(tensor, assign_m.value, assign_v.value, context, ->(m_d , v_d) {  (m_d * alpha) / (Math.sqrt(v_d) + epsilon_t) })
          assign.value = process_vector_math_op(tensor, target_var, v_delta, context, ->(var_d , delta_d) { var_d - delta_d })
          assign.value
        end

        register_op %i[softmax_cross_entropy_with_logits_v2 softmax_cross_entropy_with_logits] do |_context, tensor, inputs|
          last_dimen_list = last_axis(inputs[0])
          input_shape = shape_eval(inputs[0])
          rank = input_shape.size - 1
          labels = last_axis(inputs[1])
          func = lambda { |logits, label|
            c = logits.max
            transformed_logits = logits.map { |l| l - c }
            sum = transformed_logits.map { |x| Math.exp(x) }.reduce(:+)
            losses = transformed_logits.zip(label).map { |x, y| (Math.log(sum) - x) * y }
            probs = transformed_logits.zip(label).map  { |x, y| (Math.exp(x) / sum) - y }
            [losses, probs]
          }

          if input_shape.size == 1
            loss, prob = func.call(last_dimen_list, labels)
            loss = reduce(loss, rank, false)
            TensorStream::Evaluator::OutputGroup.new([loss, prob], [tensor.inputs[0].data_type, tensor.inputs[0].data_type])
          else
            losses = []
            backprobs = []
            arr = last_dimen_list.zip(labels).each do |list, label|
              loss, prob = func.call(list, label)
              losses << loss
              backprobs << prob
            end
            reshaped_losses = TensorShape.reshape(losses.flatten, input_shape)
            reshaped_backprops = TensorShape.reshape(backprobs.flatten, input_shape)
            reshaped_losses = reduce(reshaped_losses, rank, false)
            TensorStream::Evaluator::OutputGroup.new([reshaped_losses, reshaped_backprops], [tensor.inputs[0].data_type, tensor.inputs[0].data_type])
          end
        end

        register_op :log_softmax do |_context, _tensor, inputs|
          input_shape = shape_eval(inputs[0])
          last_dimen_list = last_axis(inputs[0])

          func = lambda { |logits|
            c = logits.max
            transformed_logits = logits.map { |l| l - c }
            sum = transformed_logits.map { |x| Math.exp(x) }.reduce(:+)
            transformed_logits.map { |x| x - Math.log(sum) }
          }

          if input_shape.size == 1
            func.call(last_dimen_list)
          else
            arr = last_dimen_list.collect do |list|
              func.call(list)
            end
            TensorShape.reshape(arr.flatten, input_shape)
          end
        end

        register_op :softmax_grad do |_context, _tensor, inputs|
          input, grad = inputs
          softmax_input = softmax(input)
          input_shape = shape_eval(input)

          last_dimen_list = last_axis(softmax_input)
          last_grad_list = last_axis(grad)

          func = lambda { |list, last_grad|
            f_grad = softmax_grad(list)
            f_grad.transpose.each.collect do |row|
              sum = 0.0
              row.each_with_index do |r, g_index|
                sum += r * last_grad[g_index]
              end
              sum
            end
          }

          if input_shape.size == 1
            func.call(last_dimen_list, last_grad_list)
          else
            arr = last_dimen_list.zip(last_grad_list).collect do |list, last_grad|
              func.call(list, last_grad)
            end
            TensorShape.reshape(arr.flatten, input_shape)
          end
        end
      end
    end
  end
end