module TensorStream
  module OpenCLHelpers
    # Collection of math functions for interfacing with OpenCL kernels
    module MathOps
      def MathOps.included(klass)
        klass.class_eval do
          %i[max min add real_div div sub floor_mod mod mul pow sigmoid_grad squared_difference].each do |op|
            register_op op, noop: true do |context, tensor, inputs|
              execute_2_operand_func(op.to_s, tensor, inputs[0], inputs[1], context)
            end
          end

          register_op :add_n do |_context, tensor, inputs|
            if inputs.size == 1
              inputs[0]
            else
              m, n = inputs[0].shape
              work_group = [m || 1, n || 1]
              cl_m = OpenCL::Int1.new(m || 1)
              cl_n = OpenCL::Int1.new(n || 1)
              cl_switch = OpenCL::Int1.new(0)
              dtype = tensor.data_type

              output_buffer = _create_result_buffer(tensor.data_type, inputs[0].shape, "out_#{tensor.name}")
              inputs_queue = inputs.dup
              a = inputs_queue.pop
              until inputs_queue.empty?
                b = inputs_queue.pop
                event_wait_list = build_event_wait_list([a, b])
                method_call = :"add_#{a.data_type}_#{b.data_type}"
                event = _cl_program('add', a: a.data_type, b: b.data_type, dtype: dtype).send(method_call, _opencl_queue, work_group, cl_m, cl_n, cl_switch, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
                a = output_buffer
                a.op = event
              end

              output_buffer.op = a.op
              output_buffer
            end
          end

          register_op :floor_div, noop: true do |context, tensor, inputs|
            if fp_type?(tensor.data_type)
              execute_2_operand_func('floor_div', tensor, inputs[0], inputs[1], context)
            else
              execute_2_operand_func('div', tensor, inputs[0], inputs[1], context)
            end
          end

          register_op :mat_mul do |_context, tensor, inputs|
            a, b = inputs

            m = a.shape[0]
            n = b.shape[1]
            v = b.shape[0]
            k = a.shape[1]

            if tensor.options[:transpose_a]
              m = a.shape[1]
              k = a.shape[0]
            end

            if tensor.options[:transpose_b]
              n = b.shape[0]
              v = b.shape[1]
            end

            result_shape = [m, n]

            raise "#{tensor.inputs[0].name} rank must be greater than 1" if a.shape.size < 2
            raise "#{tensor.inputs[1].name} rank must be greater than 1" if b.shape.size < 2
            raise "incompatible shape sizes for matrix multiplication (#{a.shape[1]} != #{b.shape[0]}) #{a.shape} vs #{b.shape}" if k != v

            dtype = tensor.data_type
            a, b = auto_type_cast(a, b, name: "#{tensor.name}/cast_#{a.name}_#{b.data_type}")
            output_buffer = _create_result_buffer(a.data_type, result_shape, tensor.name)

            cl_m = OpenCL::Int1.new(m)
            cl_n = OpenCL::Int1.new(n)
            cl_k = OpenCL::Int1.new(k)

            transpose_a = OpenCL::Int1.new(tensor.options[:transpose_a] ? 1 : 0)
            transpose_b = OpenCL::Int1.new(tensor.options[:transpose_b] ? 1 : 0)
            event_wait_list = build_event_wait_list(inputs)
            output_buffer.op = _cl_program('gemm', dtype: dtype).send(:"gemm_#{dtype}", _opencl_queue, result_shape, cl_m, cl_n, cl_k, transpose_a, transpose_b, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
            output_buffer
          end

          %i[sign exp tan acos asin sin cos abs sqrt negate square reciprocal tanh tanh_grad sigmoid log1p round floor ceil log].each do |op|
            register_op op, noop: true do |context, tensor, inputs|
              execute_func(op.to_s, tensor, inputs[0], context)
            end
          end

          %i[sum mean].each do |op|
            register_op op, noop: true do |context, tensor, inputs|
              reduction(context, tensor, inputs[0], inputs[1], op.to_sym)
            end
          end

          register_op :prod, noop: true do |context, tensor, inputs|
            input_a = complete_eval(inputs[0], context)

            if input_a.buffer.empty?
              convert_to_opencl([1.0], [], data_type: inputs[0].data_type, name: tensor.name)
            else
              reduction(context, tensor, inputs[0], inputs[1], :prod)
            end
          end

          register_op :argmin, buffer: true do |_context, tensor, inputs|
            axis = tensor.options[:axis] || 0
            rank = inputs[0].shape.size
            raise TensorStream::InvalidArgumentError, "Expected dimension in the range [#{-rank},#{rank}) but got #{axis}" if axis < -rank || axis >= rank

            arr = inputs[0].buffer.reshape(*inputs[0].shape.reverse).to_a
            op = get_op_with_axis(arr, axis, 0, inputs[0].data_type, ->(a, b) { a < b })
            convert_to_opencl(op, shape_eval(op), data_type: tensor.data_type, name: tensor.name)
          end

          register_op :argmax, buffer: true do |_context, tensor, inputs|
            axis = tensor.options[:axis] || 0
            rank = inputs[0].shape.size
            raise TensorStream::InvalidArgumentError, "Expected dimension in the range [#{-rank},#{rank}) but got #{axis}" if axis < -rank || axis >= rank

            arr = inputs[0].buffer.reshape(*inputs[0].shape.reverse).to_a
            op = get_op_with_axis(arr, axis, 0, inputs[0].data_type, ->(a, b) { a > b })
            convert_to_opencl(op, shape_eval(op), data_type: tensor.data_type, name: tensor.name)
          end
        end
      end
    end
  end
end