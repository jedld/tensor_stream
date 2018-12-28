module TensorStream
  ## Collection of machine learning related ops
  module NNOps
    def NNOps.included(klass)
      klass.class_eval do
        register_op :apply_gradient_descent do |context, tensor, inputs|
          target_var, learning_rate, delta = inputs
          assign = tensor.inputs[0] || tensor

          assign.container = process_vector_math_op(tensor, target_var, delta, context, ->(t, u) { t - u * learning_rate })
          assign.container
        end

        register_op :apply_momentum do |_context, tensor, inputs|
          target_var, momentum_var, learning_rate, grad, momentum = inputs
          assign = tensor.inputs[0] || tensor
          assign_acc = tensor.inputs[1]
          assign_acc.container = multi_array_op(->(t, u) { t * momentum + u }, momentum_var, grad)
          assign.container = if tensor.options[:use_nesterov]
                               multi_array_op(->(v, g, acc) { v - (g * learning_rate + acc * momentum * learning_rate) }, target_var, grad, momentum_var)
                             else
                               multi_array_op(->(v, acc) { v - acc * learning_rate }, target_var, momentum_var)
                             end

          assign.container
        end

        register_op :apply_adadelta do |_context, tensor, inputs|
          target_var, accum, accum_update, lr, rho, epsilon, grad = inputs
          assign = tensor.inputs[0] || tensor
          assign_acc = tensor.inputs[1]
          assign_acc_update = tensor.inputs[2]
          assign_acc.container = multi_array_op(->(acc_t, grad_t) { acc_t * rho + (grad_t * grad_t) * (1.0 - rho) }, accum, grad)
          update = multi_array_op(->(acc_update_t, acc_t, grad_t) { Math.sqrt(acc_update_t + epsilon) * (1.0 / Math.sqrt(acc_t + epsilon)) * grad_t }, accum_update, assign_acc.container, grad)
          assign.container = multi_array_op(->(v, u) { v - (u * lr) }, target_var, update)
          assign_acc_update.container = multi_array_op(->(acc_update_t, u) { acc_update_t * rho + (u * u) * (1.0 - rho) }, accum_update, update)

          assign.container
        end

        register_op :apply_adagrad do |_context, tensor, inputs|
          target_var, accum, lr, grad = inputs
          assign = tensor.inputs[0] || tensor
          assign.container = multi_array_op(->(v, a, g) { v - (g * lr * (1.0 / Math.sqrt(a))) }, target_var, accum, grad)
          assign.container
        end

        register_op :apply_adam do |_context, tensor, inputs|
          target_var, m, v, beta1_power, beta2_power, lr_t, beta1_t, beta2_t, epsilon_t, grad = inputs
          alpha = lr_t * Math.sqrt(1.0 - beta2_power) / (1.0 - beta1_power)
          assign = tensor.inputs[0]
          assign_m = tensor.inputs[1]
          assign_v = tensor.inputs[2]

          assign_m.container = multi_array_op(->(u_d , g) { u_d + (g - u_d) * (1.0 - beta1_t) }, m, grad)
          assign_v.container = multi_array_op(->(u_d , v_d) { u_d + (v_d**2 - u_d) * (1.0 - beta2_t)},  v, grad)
          assign.container = multi_array_op(->(t, m_d , v_d) { t - ((m_d * alpha) / (Math.sqrt(v_d) + epsilon_t)) }, target_var, assign_m.container, assign_v.container)
          assign.container
        end

        register_op :apply_rms_prop do |_context, tensor, inputs|
          var, ms, mom, lr, rho, momentum, epsilon, grad = inputs
          assign = tensor.inputs[0]
          assign_ms = tensor.inputs[1]
          assign_mom = tensor.inputs[2]
          assign_ms.container = multi_array_op(->(g, m) { m + (g * g - m) * (1.0 - rho)}, grad, ms)
          assign_mom.container = multi_array_op(->(mom_t, g, m) { mom_t * momentum + (g * lr) / Math.sqrt(m + epsilon)}, mom, grad, assign_ms.container)
          assign.container = multi_array_op(->(v, m) { v - m }, var, assign_mom.container)
        end

        register_op :apply_centered_rms_prop do |_context, tensor, inputs|
          var, mg, ms, mom, lr, rho, momentum, epsilon, grad = inputs
          assign = tensor.inputs[0]
          assign_mg = tensor.inputs[1]
          assign_ms = tensor.inputs[2]
          assign_mom = tensor.inputs[3]

          assign_ms.container = multi_array_op(->(g, m) { m + (g * g - m) * (1.0 - rho) }, grad, ms)
          assign_mg.container = multi_array_op(->(g, mg_t) {  (g - mg_t) * (1.0 - rho) }, grad, mg)
          denom =  multi_array_op(->(s, mg_t) { (s - mg_t * mg_t) + epsilon }, assign_ms.container, mg)
          assign_mom.container = multi_array_op(->(mom_t, g, d) { mom_t * momentum + (g * lr) / Math.sqrt(d)}, mom, grad, denom)
          assign.container = multi_array_op(->(v, m) { v - m }, var, assign_mom.container)
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
            last_dimen_list.zip(labels).each do |list, label|
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

        register_op :relu6 do |context, tensor, inputs|
          call_vector_op(tensor, :relu6, inputs[0], inputs[1], context, ->(t, u) { [[t, 0].max, 6].min })
        end

        register_op :conv2d do |_context, tensor, inputs|
          filter = inputs[1]

          filter_shape = shape_eval(filter)
          strides = tensor.options[:strides]
          padding_option = tensor.options[:padding]
          height_stride = strides[1]
          width_stride = strides[2]

          raise TensorStream::ValueError, " Current implementation does not yet support strides in the batch and depth dimensions." if strides[0] != 1 || strides[3] != 1

          _batch, height, width, _channels = shape_eval(inputs[0])
          padding = conv2d_padding_options(padding_option, filter_shape, height, width, height_stride, width_stride)
          inputs[0].collect do |image|
            f_height, f_width, _input_channels, _output_channels = filter_shape
            (-padding[0]...height).step(height_stride).map do |y|
              next if (y + f_height) > (height + padding[2])

              (-padding[1]...width).step(width_stride).map do |x|
                next if (x + f_width) > (width + padding[3])

                filter_result = (0...f_height).map do |f_y|
                  (0...f_width).map do |f_x|
                    f_element = filter[f_y][f_x]

                    next if (x + f_x >= width) || (x + f_x < 0)
                    next if (y + f_y >= height) || (y + f_y < 0)


                    image[y + f_y][x + f_x].zip(f_element).map do |image_channel, filter_channels|
                      filter_channels.map { |c| image_channel * c }
                    end
                  end.compact
                end.flatten(2)

                filter_result.transpose.map { |e| e.reduce(:+) }
              end.compact
            end.compact
          end.compact
        end

        register_op :conv2d_backprop_input do |_context, tensor, inputs|
          image_shape, filter, grad = inputs
          strides = tensor.options[:strides]
          padding_option = tensor.options[:padding]
          height_stride = strides[1]
          width_stride = strides[2]

          filter_shape = shape_eval(filter)

          f_height, f_width, _input_channels, output_channels = filter_shape
          batch, height, width, channels = image_shape

          padding = conv2d_padding_options(padding_option, filter_shape, height, width, height_stride, width_stride)

          Array.new(batch) do |b|
            image_gradient = TensorShape.reshape(Array.new(height * width * channels) { 0.0 }, [height, width, channels])

            ((0 - padding[0])...height).step(height_stride).each do |y|
              next if (y + f_height) > (height + padding[2])

              ((0 - padding[1])...width).step(width_stride).each do |x|
                next if (x + f_width) > (width + padding[3])

                (0...f_height).each do |f_y|
                  (0...f_width).each do |f_x|
                    next if (y + f_y) < 0 || (y + f_y) >= height
                    next if (x + f_x) < 0 || (x + f_x) >= width

                    img_grad = grad[b][(y + padding[0]) / height_stride][(x + padding[1]) / width_stride]

                    channels.times.each do |c|
                      g = Array.new(output_channels) do |o_c|
                        filter[f_y][f_x][c][o_c] * img_grad[o_c]
                      end.reduce(:+)

                      image_gradient[y + f_y][x + f_x][c] += g
                    end
                  end
                end

              end
            end

            image_gradient
          end
        end

        register_op :conv2d_backprop_filter do |_context, tensor, inputs|
          images, filter_shape, grad = inputs

          strides = tensor.options[:strides]
          padding_option = tensor.options[:padding]
          height_stride = strides[1]
          width_stride = strides[2]

          filter_gradient_sum = Array.new(filter_shape.reduce(:*)) { 0.0 }

          _batch, height, width, _channels = shape_eval(images)
          padding = conv2d_padding_options(padding_option, filter_shape, height, width, height_stride, width_stride)

          images.each_with_index.map do |image, index|
            f_height, f_width, input_channels, output_channels = filter_shape

            ((0 - padding[0])...height).step(height_stride).each do |y|
              ((0 - padding[1])...width).step(width_stride).each do |x|
                filter_result = (0...f_height).map do |f_y|
                  (0...f_width).map do |f_x|
                    next Array.new(input_channels * output_channels) { 0.0 } if x + f_x >= width || (x + f_x < 0) || ((x + f_width) > (width + padding[3]))
                    next Array.new(input_channels * output_channels) { 0.0 } if y + f_y >= height || (y + f_y < 0) || ((y + f_height) > (height + padding[2]))

                    image_grad = grad[index][(y + padding[0]) / height_stride][(x + padding[1])/ width_stride]
                    image[y + f_y][x + f_x].map do |image_channel|
                      Array.new(output_channels) do |o_c|
                        image_channel * image_grad[o_c]
                      end
                    end
                  end
                end.flatten

                filter_gradient_sum = multi_array_op(->(a, b) { a + b }, filter_gradient_sum, filter_result)
              end
            end
          end

          TensorShape.reshape(filter_gradient_sum, filter_shape)
        end


        def conv2d_padding_options(padding_option, filter_shape, height, width, h_stride, w_stride)
          case padding_option
          when 'SAME'
            [
              calc_pad(height, h_stride, filter_shape[0]),
              calc_pad(width, w_stride, filter_shape[1]),
              calc_pad(height, h_stride, filter_shape[0], true),
              calc_pad(width, w_stride, filter_shape[1], true)
            ]
          when 'VALID'
            [0, 0, 0, 0]
          else
            raise TensorStream::ValueError, "Unsupported padding value #{padding_option}, valid values 'SAME', 'VALID'"
          end
        end

        def calc_pad(w, stride, f_shape, ceil = false)
          r = ((w / stride - 1) * stride - w + f_shape)
          if ceil
            r.odd? ? r / 2 + 1 : r / 2
          else
            r / 2
          end
        end
      end
    end
  end
end