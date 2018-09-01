module TensorStream
  module OpenCLHelpers
    # Collection of math functions for interfacing with OpenCL kernels
    module NNOps
      def NNOps.included(klass)
        klass.class_eval do

          # Fast in place multiply subtract assign
          register_op :apply_gradient_descent do |_context, tensor, inputs|
            _target_var, learning_rate, delta = inputs

            assign = tensor.inputs[0] || tensor

            assign.buffer.dirty = true # force buffer copy when variable is read externally
            output_buffer = assign.buffer

            m, n = output_buffer.shape
            work_group = [m || 1, n || 1]
            cl_m = OpenCL::Int1.new(m || 1)
            cl_n = OpenCL::Int1.new(n || 1)

            event_wait_list = build_event_wait_list([assign.buffer, learning_rate, delta])
            method_call = :"apply_gradient_#{output_buffer.data_type}"
            event = _cl_program("apply_gradient", dtype: output_buffer.data_type).send(method_call, _opencl_queue, work_group, cl_m, cl_n, delta.cl_buffer, learning_rate.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer
          end

          # updates for gradient descent with momentum
          register_op :apply_momentum do |_context, tensor, inputs|
            target_var, momentum_var, learning_rate, grad, momentum = inputs

            assign = tensor.inputs[0] || tensor
            assign_acc = tensor.inputs[1]
            assign.buffer.dirty = true # force buffer copy when variable is read externally
            assign_acc.buffer.dirty = true # force buffer copy when variable is read externally

            output_buffer = assign.buffer

            m, n = output_buffer.shape
            work_group = [m || 1, n || 1]
            cl_m = OpenCL::Int1.new(m || 1)
            cl_n = OpenCL::Int1.new(n || 1)

            event_wait_list = build_event_wait_list([assign.buffer, assign_acc.buffer, learning_rate, grad, momentum])
            method_call = :"apply_momentum_#{output_buffer.data_type}"
            event = _cl_program("apply_momentum", nesterov: tensor.options[:use_nesterov], dtype: output_buffer.data_type).
                        send(method_call, _opencl_queue, work_group, cl_m, cl_n, grad.cl_buffer,
                            learning_rate.cl_buffer, momentum.cl_buffer, output_buffer.cl_buffer,
                            assign_acc.buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            assign_acc.buffer.op = event
            output_buffer
          end

          # Adam optimization algorithm
          register_op :apply_adam do |_context, tensor, inputs|
            _target_var, _m, _v, beta1_power, beta2_power, lr_t, beta1_t, beta2_t, epsilon_t, grad = inputs

            assign = tensor.inputs[0] || tensor
            assign_m = tensor.inputs[1]
            assign_v = tensor.inputs[2]

            # mark variable buffers as dirty
            assign.buffer.dirty = true # force buffer copy when variable is read externally
            assign_m.buffer.dirty = true # force buffer copy when variable is read externally
            assign_v.buffer.dirty = true # force buffer copy when variable is read externally

            output_buffer = assign.buffer

            m, n = output_buffer.shape
            work_group = [m || 1, n || 1]
            cl_m = OpenCL::Int1.new(m || 1)
            cl_n = OpenCL::Int1.new(n || 1)

            event_wait_list = build_event_wait_list(inputs)
            method_call = :"apply_adam_#{output_buffer.data_type}"
            event = _cl_program("apply_adam", dtype: output_buffer.data_type)
                                .send(method_call, _opencl_queue, work_group, cl_m, cl_n,
                                      grad.cl_buffer,
                                      lr_t.cl_buffer,
                                      beta1_power.cl_buffer,
                                      beta2_power.cl_buffer,
                                      beta1_t.cl_buffer,
                                      beta2_t.cl_buffer,
                                      epsilon_t.cl_buffer,
                                      assign_m.buffer.cl_buffer,
                                      assign.buffer.cl_buffer,
                                      assign_v.buffer.cl_buffer,
                                      event_wait_list: event_wait_list)
            output_buffer.op = event
            assign_m.buffer.op = event
            assign_v.buffer.op = event
            output_buffer
          end

          register_op :softmax do |_context, tensor, inputs|
            a = inputs[0]
            event_wait_list = build_event_wait_list(inputs)
            dtype = tensor.data_type
            output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)

            m, n = a.shape
            work_group = [m]
            n = m if n.nil?
            cl_n = OpenCL::Int1.new(n || 1)

            event = _cl_program("softmax", dtype: dtype).send(:"softmax_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer
          end

          register_op :log_softmax do |_context, tensor, inputs|
            a = inputs[0] # logits
            event_wait_list = build_event_wait_list(inputs)
            dtype = tensor.data_type
            output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)

            m, n = a.shape
            work_group = [m]
            n = m if n.nil?
            cl_n = OpenCL::Int1.new(n || 1)

            event = _cl_program("log_softmax", dtype: dtype).send(:"log_softmax_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer
          end

          register_op :softmax_cross_entropy_with_logits_v2 do |context, tensor, inputs|
            a = inputs[0] # logits
            b = inputs[1] # labels
            event_wait_list = build_event_wait_list(inputs)
            dtype = tensor.data_type
            output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)
            output_buffer_backprop = _create_result_buffer(tensor.data_type, a.shape, "#{tensor.name}_2")
            rank = a.shape.size - 1
            m, n = a.shape
            work_group = [m]
            n = m if n.nil?
            cl_n = OpenCL::Int1.new(n || 1)

            event = _cl_program("softmax_cross", dtype: dtype).send(:"softmax_cross_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer, b.cl_buffer,
                                 output_buffer.cl_buffer, output_buffer_backprop.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer_backprop.op = event

            loss = reduction(context, tensor, output_buffer, rank, :sum)
            TensorStream::Evaluator::OutputGroup.new([loss, output_buffer_backprop],  [tensor.inputs[0].data_type, tensor.inputs[0].data_type])
          end

          register_op :softmax_cross_entropy_with_logits_v2_grad do |_context, tensor, inputs|
            a = inputs[0] # logits
            b = inputs[1] # labels
            c = inputs[2] # grads
            event_wait_list = build_event_wait_list(inputs)
            dtype = tensor.data_type
            output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)

            m, n = a.shape
            work_group = [m]
            n = m if n.nil?
            cl_n = OpenCL::Int1.new(n || 1)

            event = _cl_program("softmax_cross_grad", dtype: dtype).send(:"softmax_cross_grad_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer, b.cl_buffer, c.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer
          end

          register_op :softmax_grad do |_context, tensor, inputs|
            a, grad = inputs

            event_wait_list = build_event_wait_list(inputs)
            dtype = tensor.data_type
            output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)

            m, n = a.shape
            work_group = [m]
            n = m if n.nil?
            cl_n = OpenCL::Int1.new(n || 1)
            event = _cl_program('softmax_grad', dtype: dtype, size: n).send(:"softmax_grad_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer, grad.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer.op = event
            output_buffer
          end
        end
      end
    end
  end
end