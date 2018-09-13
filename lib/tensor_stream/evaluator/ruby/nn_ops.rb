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
          assign_acc.value = multi_array_op(->(t, u) { t * momentum + u }, momentum_var, grad)
          if tensor.options[:use_nesterov]
            assign.value = multi_array_op(->(v, g, acc) { v - (g * learning_rate + acc * momentum * learning_rate) }, target_var, grad, momentum_var)
          else
            assign.value = multi_array_op(->(v, acc) { v - acc * learning_rate }, target_var, momentum_var)
          end
          assign.value
        end

        register_op :apply_adadelta do |_context, tensor, inputs|
          target_var, accum, accum_update, lr, rho, epsilon, grad = inputs
          assign = tensor.inputs[0] || tensor
          assign_acc = tensor.inputs[1]
          assign_acc_update = tensor.inputs[2]
          assign_acc.value = multi_array_op(->(acc_t, grad_t) { acc_t * rho + (grad_t * grad_t) * (1.0 - rho) }, accum, grad)
          update = multi_array_op(->(acc_update_t, acc_t, grad_t) { Math.sqrt(acc_update_t + epsilon) * (1.0 / Math.sqrt(acc_t + epsilon)) * grad_t }, accum_update, assign_acc.value, grad)
          assign.value = multi_array_op(->(v, u) { v - (u * lr) }, target_var, update)
          assign_acc_update.value = multi_array_op(->(acc_update_t, u) { acc_update_t * rho + (u * u) * (1.0 - rho) }, accum_update, update)

          assign.value
        end

        register_op :apply_adagrad do |_context, tensor, inputs|
          target_var, accum, lr, grad = inputs
          assign = tensor.inputs[0] || tensor

          assign.value = multi_array_op(->(v, a, g) { v - (g * lr * (1.0 / Math.sqrt(a))) }, target_var, accum, grad)
          assign.value
        end

        register_op :apply_adam do |_context, tensor, inputs|
          target_var, m, v, beta1_power, beta2_power, lr_t, beta1_t, beta2_t, epsilon_t, grad = inputs
          alpha = lr_t * Math.sqrt(1.0 - beta2_power) / (1.0 - beta1_power)
          assign = tensor.inputs[0]
          assign_m = tensor.inputs[1]
          assign_v = tensor.inputs[2]

          assign_m.value = multi_array_op(->(u_d , g) { u_d + (g - u_d) * (1.0 - beta1_t) }, m, grad)
          assign_v.value = multi_array_op(->(u_d , v_d) { u_d + (v_d**2 - u_d) * (1.0 - beta2_t)},  v, grad)
          assign.value = multi_array_op(->(t, m_d , v_d) { t - ((m_d * alpha) / (Math.sqrt(v_d) + epsilon_t)) }, target_var, assign_m.value, assign_v.value)
          assign.value
        end

        register_op :apply_rms_prop do |_context, tensor, inputs|
          var, ms, mom, lr, rho, momentum, epsilon, grad = inputs
          assign = tensor.inputs[0]
          assign_ms = tensor.inputs[1]
          assign_mom = tensor.inputs[2]
          assign_ms.value = multi_array_op(->(g, m) { m + (g * g - m) * (1.0 - rho)}, grad, ms)
          assign_mom.value = multi_array_op(->(mom_t, g, m) { mom_t * momentum + (g * lr) / Math.sqrt(m + epsilon)}, mom, grad, assign_ms.value)
          assign.value = multi_array_op(->(v, m) { v - m }, var, assign_mom.value)
        end

        register_op :apply_centered_rms_prop do |_context, tensor, inputs|
          var, mg, ms, mom, lr, rho, momentum, epsilon, grad = inputs
          assign = tensor.inputs[0]
          assign_mg = tensor.inputs[1]
          assign_ms = tensor.inputs[2]
          assign_mom = tensor.inputs[3]

          assign_ms.value = multi_array_op(->(g, m) { m + (g * g - m) * (1.0 - rho) }, grad, ms)
          assign_mg.value = multi_array_op(->(g, mg_t) {  (g - mg_t) * (1.0 - rho) }, grad, mg)
          denom =  multi_array_op(->(s, mg_t) { (s - mg_t * mg_t) + epsilon }, assign_ms.value, mg)
          assign_mom.value = multi_array_op(->(mom_t, g, d) { mom_t * momentum + (g * lr) / Math.sqrt(d)}, mom, grad, denom)
          assign.value = multi_array_op(->(v, m) { v - m }, var, assign_mom.value)
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

        register_op :sparse_softmax_cross_entropy_with_logits do |context, tensor, inputs|
          last_dimen_list = last_axis(inputs[0])
          input_shape = shape_eval(inputs[0])
          rank = input_shape.size - 1
          labels = last_axis(inputs[1])
          num_classes = input_shape.last

          labels = labels.map do |l|
            one_hot = Array.new(num_classes) { 0 }
            one_hot[l] = 1
            one_hot
          end

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
            reshaped_losses = TensorShape.reshape(losses, input_shape)
            reshaped_backprops = TensorShape.reshape(backprobs, input_shape)
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
            TensorShape.reshape(arr, input_shape)
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
            TensorShape.reshape(arr, input_shape)
          end
        end
      end
    end
  end
end