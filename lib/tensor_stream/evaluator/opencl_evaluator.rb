require 'tensor_stream/evaluator/operation_helpers/random_gaussian'
require 'tensor_stream/evaluator/operation_helpers/array_ops_helper'
require 'tensor_stream/evaluator/operation_helpers/math_helper'
require 'tensor_stream/evaluator/opencl_buffer'
require 'tensor_stream/evaluator/opencl_template_helper'
require 'opencl_ruby_ffi'
require 'narray_ffi'
require 'tensor_stream/evaluator/base_evaluator'

module TensorStream
  module Evaluator
    class FullEvalNotPossible < RuntimeError
    end

    # Errors during graph evaluation
    class EvaluatorExcecutionException < RuntimeError
      attr_reader :tensor

      def initialize(exception, tensor)
        @exception = exception
        @tensor = tensor
      end

      def wrapped_exception
        @exception
      end
    end

    ## PURE ruby evaluator used for testing and development
    class OpenclEvaluator < BaseEvaluator
      attr_accessor :retain

      include TensorStream::OpHelper
      include TensorStream::ArrayOpsHelper
      include TensorStream::MathHelper

      def initialize(session,thread_pool: nil, log_intermediates: false)
        super
      end

      def self.query_supported_devices
        devices = query_devices_with_score
        devices.sort { |a| a[1] }.reverse.map do |d|
          device = d[0]
          index = d[3]
          platform_name = device.platform.name.gsub(/(.)([A-Z])/,'\1_\2').downcase
          uri = [platform_name, index].join('/')

          Device.new(uri, device.type, 'opencl')
        end
      end

      # opencl evaluator main entrypoint
      def run(tensor, execution_context)
        read_final_result(complete_eval(tensor, execution_context))
      end

      def run_with_buffer(tensor, context, execution_context)
        @context = context
        @context[:_cache][:_cl_buffers] ||= {} if context[:_cache]
        _create_opencl_context
        create_command_queue

        if tensor.is_a?(Array)
          tensor.collect do |t|
            value = run(t, execution_context)
            Buffer.new(data_type: t.data_type, buffer: value)
          end
        else
          value = run(tensor, execution_context)
          Buffer.new(data_type: tensor.data_type, buffer: value)
        end
      end

      def convert_from_buffer(tensor, result)
        convert_to_opencl([result.buffer].flatten, shape_eval(result.buffer), data_type: result.data_type, name: tensor.name)
      end

      def complete_eval(tensor, context)
        buffer = _run(tensor, context)
        if buffer.is_a?(Array)
          buffer = buffer.collect do |b|
            next b if b.buffer.size.zero?
            _opencl_queue.enqueue_read_buffer(b.cl_buffer, b.buffer, event_wait_list: [b.op].compact)
            b
          end
        else
          return buffer if buffer.nil? || buffer.buffer.size.zero?
          _opencl_queue.enqueue_read_buffer(buffer.cl_buffer, buffer.buffer, event_wait_list: [buffer.op].compact)
        end
        _opencl_queue.finish
        buffer
      end

      def opencl_device
        @context[:_cache][:_opencl_device]
      end

      protected

      def prepare_input(tensor, context, options = {})
        return nil unless tensor
        tensor = resolve_placeholder(tensor)
        if options[:noop]
          tensor
        elsif options[:buffer]
          complete_eval(tensor, context)
        elsif options[:complete]
          read_final_result(complete_eval(tensor, context))
        else
          _run(tensor, context)
        end
      end

      # read result from opencl and convert to ruby
      def read_final_result(buffer)
        return buffer.map { |b| read_final_result(b) } if buffer.is_a?(Array)
        return nil if buffer.nil?

        buffer.to_ruby
      end

      def _create_opencl_context
        @context[:_cache][:_opencl_device] ||= begin
          if @preferred_device
            @preferred_device
          else
            device, _score, _platform, _index = choose_best_device
            # puts "using #{device.name}"
            device
          end
        end
        @context[:cl_device] = opencl_device
        @context[:_cache][:_opencl_context] ||= OpenCL.create_context(opencl_device)
      end

      def choose_best_device
        @best_device ||= begin
          devices = OpenclEvaluator.query_devices_with_score
          devices.sort { |a| a[1] }.reverse.first
        end
      end

      def self.query_devices_with_score
        OpenCL.platforms.flat_map do |p|

          p.devices.select { |d| d.available > 0 }.each_with_index.collect do |d, index|
            score = 0
            if d.type.to_s == 'CPU'
              score += 1
            elsif d.type.to_s == 'GPU'
              score += 4
            end

            if d.platform.name == 'NVIDIA CUDA'
              score += 1000
            end

            score += d.max_compute_units
            score += d.max_clock_frequency

            [d, score, p.name, index]
          end
        end
      end

      def create_command_queue
        supported_proprties = opencl_device.queue_properties.names
        properties = []
        properties << OpenCL::CommandQueue::PROFILING_ENABLE if supported_proprties.include?('PROFILING_ENABLE')
        properties << OpenCL::CommandQueue::OUT_OF_ORDER_EXEC_MODE_ENABLE if supported_proprties.include?('OUT_OF_ORDER_EXEC_MODE_ENABLE')
        @context[:_cache][:_opencl_queue] ||= _opencl_context.create_command_queue(opencl_device, properties: properties)
      end

      def _opencl_context
        @context[:_cache][:_opencl_context]
      end

      def _opencl_queue
        @context[:_cache][:_opencl_queue]
      end

      def cl_template_path(kernel, extension)
        File.join(File.dirname(__FILE__), 'kernels', "#{kernel}.#{extension}")
      end

      def _cl_program(kernel, args = {})
        suffix = args.collect { |k,v| "#{k}.#{v}"}.join('.')
        @context[:_cache]["_opencl_kernel_#{kernel}.#{suffix}"] ||= begin
          filename = %w[cl.erb cl].map { |ext| cl_template_path(kernel, ext) }.find { |n| File.exist?(n) }
          source = File.read(filename)
          source = OpenclTemplateHelper.new(source).generate(args)
          # File.write("/tmp/#{kernel}.#{suffix}.cl", source)
          program = _opencl_context.create_program_with_source(source)
          program.build
        rescue OpenCL::Error::BUILD_PROGRAM_FAILURE => e
          puts "OpenCL Compile error: #{program.build_log}"
          raise e
        end
      end

      def _run(tensor, execution_context)
        return tensor if tensor.is_a?(OpenCLBuffer)
        if tensor.is_a?(Array) && tensor.size > 0 && tensor[0].is_a?(Tensor)
          return tensor.map { |t| _run(t, execution_context) }
        end

        tensor = tensor.call if tensor.is_a?(Proc)

        child_context = execution_context.dup
        res = if tensor.is_a?(Operation)
                if !self.class.ops.include?(tensor.operation.to_sym)
                  result = @session.delegate_to_evaluator(tensor, @context, execution_context)
                  convert_from_buffer(tensor, result)
                else
                  eval_operation(tensor, child_context)
                end
              elsif tensor.is_a?(Variable)
                eval_variable(tensor, child_context)
              elsif tensor.is_a?(Placeholder)
                resolve_placeholder(tensor, child_context)
              else
                eval_tensor(tensor, child_context)
              end
        execution_context.deep_merge!(returns: child_context[:returns])
        res
      end

      def eval_variable(tensor, child_context)
        raise "variable #{tensor.name} not initalized" if tensor.value.nil? && (tensor.buffer.nil? || !tensor.buffer.dirty)
        tensor.buffer = wrap_opencl(tensor, name: tensor.name) if tensor.buffer.nil?
        tensor.buffer
      end

      register_op :log do |context, tensor, inputs|
        execute_func('log', tensor, inputs[0], context)
      end

      register_op :sin do |context, tensor, inputs|
        execute_func('sin', tensor, inputs[0], context)
      end

      register_op :cond do |context, tensor, inputs|
        pred = complete_eval(tensor.options[:pred], context)

        if all_true?(pred.buffer)
          inputs[0]
        else
          inputs[1]
        end
      end

      register_op :identity do |_context, _tensor, inputs|
        inputs[0]
      end

      register_op :assign, noop: true do |context, tensor, inputs|
        assign_var(tensor, inputs[1], context)
      end

      register_op :assign_add do |context, tensor, inputs|
        value = execute_2_operand_func('add', tensor, inputs[0], inputs[1], context)
        assign_var(tensor, value, context)
      end

      register_op :assign_sub do |context, tensor, inputs|
        value = execute_2_operand_func('sub', tensor, inputs[0], inputs[1], context)
        assign_var(tensor, value, context)
      end

      %i[less less_equal greater greater_equal equal not_equal logical_and].each do |op|
        register_op op, noop: true do |context, tensor, inputs|
          execute_2_operand_func(op.to_s, tensor, inputs[0], inputs[1], context, 'cond')
        end
      end

      %i[max add div sub mul pow sigmoid_grad].each do |op|
        register_op op, noop: true do |context, tensor, inputs|
          execute_2_operand_func(op.to_s, tensor, inputs[0], inputs[1], context)
        end
      end

      register_op :where, noop: true do |context, tensor, inputs|
        pred = tensor.options[:pred]
        execute_cond_func('where', tensor, pred, inputs[0], inputs[1], context)
      end

      register_op :matmul do |_context, tensor, inputs|
        a, b = inputs

        m = a.shape[0]
        n = b.shape[1]
        v = b.shape[0]
        k = a.shape[1]

        m, k = [a.shape[1], a.shape[0]] if tensor.options[:transpose_a]
        n, v = [b.shape[0], b.shape[1]] if tensor.options[:transpose_b]

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

        output_buffer.op = _cl_program('gemm', dtype: dtype).send(:"gemm_#{dtype}", _opencl_queue, result_shape, cl_m, cl_n, cl_k, transpose_a, transpose_b, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer)
        output_buffer
      end

      register_op :cast do |_context, tensor, inputs|
        a = inputs[0]
        if a.data_type != tensor.data_type
          buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)
          m, n = a.shape
          cl_m = OpenCL::Int1.new(m || 1)
          cl_n = OpenCL::Int1.new(n || 1)
          work_group = [m || 1, n || 1]

          buffer.op = _cl_program("cast", source_dt: a.data_type, target_dt: tensor.data_type).cast(_opencl_queue, work_group, cl_m, cl_n, a.cl_buffer, buffer.cl_buffer)
          buffer
        else
          a
        end
      end

      %i[sign exp tan cos abs sqrt negate square reciprocal tanh tanh_grad sigmoid log1p round].each do |op|
        register_op op, noop: true do |context, tensor, inputs|
          execute_func(op.to_s, tensor, inputs[0], context)
        end
      end

      register_op :softmax do |_context, tensor, inputs|
        a = inputs[0]
        event_wait_list = [a.op].compact
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

      register_op :softmax_grad do |_context, tensor, inputs|
        a, grad = inputs

        event_wait_list = [a.op].compact
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

      register_op :truncate do |context, tensor, inputs|
        a, b = inputs
        if a.shape.size.zero?
          a
        else
          input_b = read_final_result(b)
          if a.shape == input_b
            a
          else
            input_a = read_final_result(a)
            if input_b == []
              if a.buffer.size == 1
                a.shape = input_b
                a
              else
                wrap_opencl(a.buffer[0], data_type: a.data_type, name: tensor.name)
              end
            else
              wrap_opencl(truncate(input_a, input_b), data_type: a.data_type, name: tensor.name)
            end
          end
        end
      end

      register_op :check_numerics, noop: true do |context, tensor, inputs|
        a = complete_eval(inputs[0], context)
        name = tensor.options[:name]

        a.buffer.each do |input|
          raise "#{name} Invalid Argument" if input.nan? || input.infinite?
        end
        a
      end

      register_op :broadcast_transform do |context, tensor, inputs|
        a, b = inputs

        if a.shape == b.shape
          [a, b]
        else
          input_a = read_final_result(complete_eval(a, context))
          input_b = read_final_result(complete_eval(b, context))
          b_a, b_b = broadcast(input_a, input_b)
          [ wrap_opencl(b_a, data_type: a.data_type, name: "#{tensor.name}_a"),
            wrap_opencl(b_b, data_type: a.data_type, name: "#{tensor.name}_b")]
        end
      end

      register_op :print do |context, tensor, inputs|
        a, b = inputs
        input_b = complete_eval(b, context)
        input_b = read_final_result(input_b)
        puts "#{tensor.options.fetch(:message, '')} #{input_b}"
        a
      end

      register_op :rank do |_context, tensor, inputs|
        wrap_opencl(inputs[0].shape.size, data_type: tensor.data_type, name: tensor.name)
      end

      register_op :stop_gradient do |_context, _tensor, inputs|
        inputs[0]
      end

      register_op :slice, noop: true do |context, tensor, inputs|
        input_a = complete_eval(inputs[0], context)
        input_b = read_final_result(complete_eval(inputs[1], context))
        size = tensor.options[:size]

        slice_param = input_b.zip(size).collect { |x, y| x..x + y - 1 }.reverse

        new_buf = input_a.buffer.reshape(*input_a.shape.reverse)
        sliced = new_buf.slice[*slice_param]
        convert_to_opencl(sliced.flatten, sliced.shape.reverse, data_type: inputs[0].data_type, name: tensor.name)
      end

      register_op :transpose, buffer: true do |_context, tensor, inputs|
        t_param = Array.new(inputs[0].shape.size) { |index| index }.reverse
        transposed = inputs[0].buffer.reshape(*inputs[0].shape.reverse).transpose(*t_param)
        convert_to_opencl(transposed.flatten, transposed.shape.reverse, data_type: inputs[0].data_type, name: tensor.name)
      end

      register_op :index, buffer: true do |_context, tensor, inputs|
        a = inputs[0]
        input_a = read_final_result(a)
        index = read_final_result(inputs[1])

        if a.is_a?(Array)
          a[index]
        else
          new_shape = a.shape.dup
          new_shape.shift
          convert_to_opencl(input_a[index], new_shape, data_type: a.data_type, name: tensor.name)
        end
      end

      register_op :broadcast_gradient_args, buffer: true do |_context, tensor, inputs|
        wrap_opencl(get_broadcast_gradient_args(inputs[0].buffer.to_a, inputs[1].buffer.to_a), data_type: inputs[0].data_type, name: tensor.name)
      end

      register_op :shape do |_context, tensor, inputs|
        wrap_opencl(inputs[0].shape, name: tensor.name, data_type: tensor.options[:out_type] || :float32)
      end

      register_op :reshape, buffer: true do |_context, _tensor, inputs|
        arr = inputs[0]
        new_shape = read_final_result(inputs[1])

        if new_shape.size.zero? && arr.buffer.size == 1
          arr.shape = new_shape
          arr
        else
          new_shape = TensorShape.fix_inferred_elements(new_shape, arr.buffer.size)
          arr.shape = new_shape
          arr
        end
      end

      register_op :flow_group do |_context, _tensor, inputs|
        inputs
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
        arr = inputs[0].buffer.reshape(*inputs[0].shape.reverse).to_a
        op = get_op_with_axis(arr, axis, 0, inputs[0].data_type, ->(a, b) { a < b })
        convert_to_opencl(op, shape_eval(op), data_type: tensor.data_type, name: tensor.name)
      end

      register_op :argmax, buffer: true do |_context, tensor, inputs|
        axis = tensor.options[:axis] || 0
        arr = inputs[0].buffer.reshape(*inputs[0].shape.reverse).to_a
        op = get_op_with_axis(arr, axis, 0, inputs[0].data_type, ->(a, b) { a > b })
        convert_to_opencl(op, shape_eval(op), data_type: tensor.data_type, name: tensor.name)
      end

      def eval_operation(tensor, child_context)
        return @context[tensor.name] if @context.key?(tensor.name)
        cache_key = "#{tensor.graph.object_id}_opencl_#{tensor.name}"
        return @context[cache_key] if @context.key?(cache_key)

        a = resolve_placeholder(tensor.inputs[0], child_context) if tensor.inputs && tensor.inputs[0]
        b = resolve_placeholder(tensor.inputs[1], child_context) if tensor.inputs && tensor.inputs[1]
        # puts tensor.name
        invoke(tensor, child_context).tap do |result|
          # puts "#{tensor.to_math(true,1)} = #{read_final_result(complete_eval(result, child_context))}"
          if tensor.breakpoint
            a = read_final_result(complete_eval(a, child_context))
            b = read_final_result(complete_eval(b, child_context))
            result = read_final_result(complete_eval(result, child_context))

            tensor.breakpoint.call(tensor, a, b, result)
          end
          if @log_intermediates
            @context[:compute_history] << {
              name: tensor.name,
              type: tensor.data_type,
              shape: shape_eval(result),
              source: tensor.source,
              description: tensor.to_math(true, 1),
              value: result
            }
          end
          @context[:_cache][cache_key] =  @context[cache_key] if tensor.is_const
          @context[tensor.name] = result
        end
      rescue EvaluatorExcecutionException => e
        raise e
      rescue StandardError => e
        _opencl_queue.finish # dump queue
        puts e.message
        puts e.backtrace.join("\n")

        # shape_a = a.shape.shape if a
        # shape_b = b.shape.shape if b
        # dtype_a = a.data_type if a
        # dtype_b = b.data_type if b
        # a = complete_eval(a, child_context)
        # b = complete_eval(b, child_context)
        # puts "name: #{tensor.given_name}"
        # # puts "op: #{tensor.to_math(true, 1)}"
        # puts "A #{shape_a} #{dtype_a}: #{a}" if a
        # puts "B #{shape_b} #{dtype_b}: #{b}" if b
        # dump_intermediates if @log_intermediates
        # File.write('/home/jedld/workspace/tensor_stream/samples/error.graphml', TensorStream::Graphml.new.get_string(tensor, @session))

        # File.write('/Users/josephemmanueldayo/workspace/gradients.graphml', TensorStream::Graphml.new.get_string(tensor, @session))
        raise EvaluatorExcecutionException.new(e, tensor), "error #{e.message} while evaluating #{tensor.name} : #{tensor.to_math(true,1)} defined at #{tensor.source}"
      end

      def eval_tensor(tensor, child_context)
        return tensor unless tensor.is_a?(Tensor)

        cache_key = "#{tensor.graph.object_id}_opencl_#{tensor.name}"
        return @context[cache_key] if @context.key?(cache_key)
        return @context[:_cache][cache_key] if tensor.is_const && @context[:_cache][cache_key]
        @context[cache_key] = if tensor.value.is_a?(Tensor)
                                _run(tensor.value, child_context)
                              else
                                wrap_opencl(tensor, name: tensor.name)
                              end
        @context[:_cache][cache_key] =  @context[cache_key] if tensor.is_const
      end

      private

      def assign_var(tensor, b, child_context)
        assign = tensor.inputs[0] || tensor
        buffer = complete_eval(b, child_context)

        if assign.buffer
          buffer = type_cast(buffer, assign.data_type, name: "#{tensor.name}/cast_#{tensor.name}_#{tensor.data_type}")
          if assign.buffer.cl_buffer != buffer.cl_buffer
            assign.buffer.op = _opencl_queue.enqueue_copy_buffer(buffer.cl_buffer, assign.buffer.cl_buffer, event_wait_list: [buffer.op, assign.buffer.op])
          end
        else
          assign.buffer = convert_to_opencl(read_final_result(buffer), buffer.shape, data_type: tensor.data_type, name: tensor.name)
        end
        assign.buffer.dirty = true
        assign.buffer
      end

      def execute_2_operand_func(op_name, tensor, input_a, input_b, child_context, prog_name = nil)
        a = _run(input_a, child_context)
        b = _run(input_b, child_context)
        a, b = auto_type_cast(a, b, name: "#{tensor.name}/cast_#{a.name}_#{b.data_type}")
        dtype = tensor.data_type
        result_shape = TensorShape.infer_shape(a.shape, b.shape)

        output_buffer = _create_result_buffer(tensor.data_type, result_shape, tensor.name)
        a, b, prog, switch_operands = select_program(a, b, op_name)
        m, n = result_shape
        work_group = [m || 1, n || 1]
        cl_m = OpenCL::Int1.new(m || 1)
        cl_n = OpenCL::Int1.new(n || 1)
        cl_switch = OpenCL::Int1.new(switch_operands) # no need to switch for addition

        event_wait_list = [a.op, b.op].compact # add dependency wait list

        method_call = :"#{prog}_#{a.data_type}_#{b.data_type}"
        event = if prog == "#{op_name}_b"
          cl_m_b, cl_n_b = if b.shape.size == 2
            [ OpenCL::Int1.new(b.shape[0]), OpenCL::Int1.new(b.shape[1]) ]
          elsif b.shape.size == 1
            [ OpenCL::Int1.new(1), OpenCL::Int1.new(b.shape[0]) ]
          else
            raise "rank > 2 not supported!"
          end
          _cl_program("#{prog_name || op_name}", a: a.data_type, b: b.data_type, dtype: dtype).send(method_call, _opencl_queue, work_group, cl_m, cl_n, cl_m_b, cl_n_b, cl_switch, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
        else
          _cl_program("#{prog_name || op_name}", a: a.data_type, b: b.data_type, dtype: dtype).send(method_call, _opencl_queue, work_group, cl_m, cl_n, cl_switch, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
        end

        output_buffer.op = event
        output_buffer
      end

      def execute_cond_func(op_name, tensor, pred, input_a, input_b, child_context)
        p = _run(pred, child_context)
        a = _run(input_a, child_context)
        b = _run(input_b, child_context)

        a, b = auto_type_cast(a, b, name: "#{tensor.name}/cast_#{a.name}_#{b.data_type}")
        dtype = tensor.data_type

        output_buffer = _create_result_buffer(tensor.data_type, p.shape, tensor.name)

        m, n = p.shape
        work_group = [m || 1, n || 1]
        cl_m = OpenCL::Int1.new(m || 1)
        cl_n = OpenCL::Int1.new(n || 1)

        event_wait_list = [a.op, b.op, p.op].compact # add dependency wait list
        output_buffer.op = _cl_program("#{op_name}", dtype: dtype).send(:"#{op_name}_#{dtype}", _opencl_queue, work_group, cl_m, cl_n, p.cl_buffer, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
        output_buffer
      end

      def execute_func(op_name, tensor, a, child_context)
        a = _run(a, child_context)
        event_wait_list = [a.op].compact
        dtype = tensor.data_type
        output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)

        m, n = a.shape
        work_group = [m || 1, n || 1]
        cl_m = OpenCL::Int1.new(m || 1)
        cl_n = OpenCL::Int1.new(n || 1)

        event = _cl_program("#{op_name}", dtype: dtype).send(:"#{op_name}_#{dtype}", _opencl_queue, work_group, cl_m, cl_n, a.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
        output_buffer.op = event
        output_buffer
      end

      def auto_type_cast(a, b, name: nil)
        return [a, b] if a.data_type == b.data_type
        m, n = b.shape
        work_group = [m || 1, n || 1]
        event_wait_list = [b.op].compact
        buffer = _create_result_buffer(b.data_type, b.shape, name)

        cl_m = OpenCL::Int1.new(m || 1)
        cl_n = OpenCL::Int1.new(n || 1)

        buffer.op = _cl_program("cast", source_dt: a.data_type, target_dt: b.data_type).cast(_opencl_queue, work_group, cl_m, cl_n, b.cl_buffer, buffer.cl_buffer, event_wait_list: event_wait_list)
        [a, buffer]
      end

      def type_cast(source, data_type, name: nil)
        return source if source.data_type == data_type
        m, n = source.shape
        work_group = [m || 1, n || 1]
        event_wait_list = [source.op].compact
        buffer = _create_result_buffer(data_type, source.shape, name)

        cl_m = OpenCL::Int1.new(m || 1)
        cl_n = OpenCL::Int1.new(n || 1)

        buffer.op = _cl_program("cast", source_dt: source.data_type, target_dt: data_type).cast(_opencl_queue, work_group, cl_m, cl_n, source.cl_buffer, buffer.cl_buffer, event_wait_list: event_wait_list)
        buffer
      end

      def wrap_opencl(tensor, data_type: nil, name: nil)
        value, shape = if tensor.is_a?(Tensor)
          [tensor.value, tensor.shape.shape]
        else
          [tensor , shape_eval(tensor)]
        end

        convert_to_opencl(value, shape, data_type: data_type || tensor.data_type, name: name)
      end

      def convert_to_opencl(value, shape, data_type: nil, name: nil)
        if !value.is_a?(Array) && !value.is_a?(NArray)
          value = [value]
        end

        cache_key = "_cl_object_#{name}:#{shape.join('_')}"
        cl_object =  if name && @context[:_cache][cache_key]
                      @context[:_cache][cache_key]
                     else
                       narray_size = shape.reduce(:*) || 1

                       buffer = if value.is_a?(NArray)
                                  value
                                else
                                  allocate_narray_for_type(data_type, narray_size)
                                end

                       cl_buffer_size = shape.empty? ? 1 : shape.reduce(:*)

                       cl_buffer = if !value.flatten.empty?
                        cl_buffer_size = 1 if cl_buffer_size.zero?
                        _opencl_context.create_buffer(cl_buffer_size * buffer.element_size)
                       else
                        nil
                       end

                       @context[:_cache][cache_key] = OpenCLBuffer.new(name: name, data_type: data_type, shape: shape, buffer: buffer, cl_buffer: cl_buffer)
                     end

        if value.is_a?(Array)
          value.flatten.each_with_index do |element, index|
            if element.is_a?(Tensor)
              cl_object.buffer[index] = read_final_result(complete_eval(element, {}))
            else
              cl_object.buffer[index] = ( data_type == :boolean ? ( element ? 1 : 0 ) : Tensor.cast_dtype(element, data_type))
            end
          end
        elsif value.is_a?(NArray)
          cl_object.buffer = value
        else
          cl_object.buffer[0] = ( data_type == :boolean ? ( element ? 1 : 0 )  : Tensor.cast_dtype(value, data_type))
        end

        write_op = if cl_object.cl_buffer && !value.nil? && (!value.is_a?(Array) || !value.empty?)
          _opencl_queue.enqueue_write_buffer(cl_object.cl_buffer, cl_object.buffer)
        end
        cl_object.op = write_op
        cl_object
      end

      def allocate_narray_for_type(data_type, narray_size)
        case data_type
        when :float, :float32
          NArray.sfloat(narray_size)
        when :float64
          NArray.float(narray_size)
        when :int, :int32, :int64
          NArray.int(narray_size)
        when :int16
          NArray.sint(narray_size)
        when :boolean
          NArray.sint(narray_size)
        else
          raise "unsupported type #{data_type}"
        end
      end

      def _create_result_buffer(data_type, shape, name)
        @context[:_cache][:_cl_buffers]["_result_#{name}_#{shape.join('_')}"] ||= begin
          size = shape.empty? ? 1 : shape.reduce(:*)
          buffer =  allocate_narray_for_type(data_type, size)
          cl_buffer = _opencl_context.create_buffer(buffer.size * buffer.element_size)
          OpenCLBuffer.new(data_type: data_type, shape: shape, buffer: buffer, cl_buffer: cl_buffer)
        end
      end

      def get_op_with_axis(a, target_axis, current_axis, output_type, op = ->(t, u) { t > u })
        if target_axis == current_axis
          if a[0].is_a?(Array)
            (0...a[0].size).each.collect do |column_index|
              max = nil
              max_index = 0
              a.each_with_index do |row, row_index|
                if max.nil? || op.call(row[column_index], max)
                  max = row[column_index]
                  max_index = row_index
                end
              end

              Tensor.cast_dtype(max_index, output_type)
            end
          else
            max = nil
            max_index = 0
            a.each_with_index do |x, index|
              if max.nil? || op.call(x, max)
                max = x
                max_index = index
              end
            end
            Tensor.cast_dtype(max_index, output_type)
          end
        else
          a.collect do |row|
            get_op_with_axis(row, target_axis, current_axis + 1, output_type, op)
          end
        end
      end

      def reduction(child_context, tensor, a, b, func)
        input = complete_eval(a, child_context)
        axis = read_final_result(complete_eval(b, child_context))
        if axis.nil?
          red = input.buffer.send(func)
          convert_to_opencl(red, [], data_type: tensor.data_type, name: tensor.name)
        else
          return input if input.shape.empty?
          value = input.buffer.reshape(*input.shape.reverse)
          rank = input.shape.size - 1

          if axis.is_a?(Array)
            axis.map{ |x| rank - x.abs }.sort.reverse.each do |x|
              value = value.send(func, x)
            end
          else
            value = value.send(func, rank - axis.abs)
          end

          new_shape = if value.is_a?(NArray)
            value.shape.reverse
          else
            value = [value]
            []
          end

          if tensor.options[:keepdims]
            new_shape = reduced_shape(input.shape.dup, axis)
          end

          convert_to_opencl(value.flatten, new_shape, data_type: tensor.data_type, name: tensor.name)
        end
      end

      def arr_pad(arr, paddings, data_type = :float32, rank = 0)
        raise "padding #{paddings[rank]} needs to have to elements [before, after]" if paddings[rank].size != 2

        before = paddings[rank][0]
        after = paddings[rank][1]
        pad_value = fp_type?(data_type) ? 0.0 : 0
        if arr[0].is_a?(Array)
          next_dim_elem = arr.collect { |a| arr_pad(a, paddings, data_type, rank + 1) }
          padding = deep_dup_array(next_dim_elem[0], pad_value)
          Array.new(before) { padding } + next_dim_elem + Array.new(after) { padding }
        else
          Array.new(before) { pad_value } + arr + Array.new(after) { pad_value }
        end
      end

      def deep_dup_array(arr, value = nil)
        if arr.is_a?(Array)
          arr.dup.collect do |a|
            deep_dup_array(a, value)
          end
        else
          value.nil? ? arr : value
        end
      end

      def matmul_const_transform(mat, mat_b, tensor)
        if !mat.is_a?(Array)
          compat_shape = shape_eval(mat_b).reverse
          func = -> { tensor.data_type == :int32 ? mat.to_i : mat.to_f }

          generate_vector(compat_shape, generator: func)
        else
          mat
        end
      end

      # determine possible reduction axis to be used
      def _broadcast_gradient_op(vector_shape1, vector_shape2, level)
        va_rank = _rank_from_shape(vector_shape1)
        vb_rank = _rank_from_shape(vector_shape2)
        return [] if vector_shape1 == vector_shape2 # same shape so no reductions

        shape2_r = vector_shape2.reverse

        vector_shape1.reverse.each_with_index.collect do |s, index|
          next va_rank - index - 1 if index >= shape2_r.size
          next nil if shape2_r[index] == s
          next nil if shape2_r[index] > s
          va_rank - index - 1
        end.compact
      end

      # selects variants of cl programs depending on input
      def select_program(input_a, input_b, op)
        return [input_a, input_b, "#{op}", 0] if input_a.shape == input_b.shape

        return [input_b, input_a, "#{op}_c", 1] if input_a.shape.empty? || input_a.shape.reduce(:*) == 1 # A is scalar?
        return [input_a, input_b, "#{op}_c", 0] if input_b.shape.empty? || input_a.shape.reduce(:*) == 1 # B is scalar?

        return [input_b, input_a, "#{op}_b", 1] if input_a.shape.size < input_b.shape.size

        if input_a.shape.size == input_b.shape.size
          input_a.shape.zip(input_b.shape).each do |s1, s2|
            return [input_b, input_a, "#{op}_b", 1] if s1 < s2
          end
        end

        [input_a, input_b, "#{op}_b", 0]
      end

      def _rank_from_shape(shape)
        shape.is_a?(Array) ? shape.size : 0
      end

      def get_broadcast_gradient_args(input_a, input_b)
        return [] if get_rank(input_b).zero? && get_rank(input_a).zero?
        return nil if get_rank(input_b).zero?
        # ruby scalar
        if get_rank(input_a).zero?
          _broadcast_gradient_op(input_b, input_a, 0, true)
        elsif get_rank(input_a) > 0
          _broadcast_gradient_op(input_a, input_b, 0)
        end
      end

      def concat_array(values, axis)
        combined_array = values.shift
        axis = get_rank(combined_array) - 1 if axis == -1

        values.each do |v|
          combined_array = concat(combined_array, v, axis)
        end
        combined_array
      end

      def concat(a, b, axis)
        if axis.zero?
          a + b
        else
          a.each_with_index.collect do |i, index|
            concat(i, b[index], axis - 1)
          end
        end
      end

      def resolve_placeholder(placeholder, _execution_context = {})
        return nil if placeholder.nil?

        var = if placeholder.is_a?(Placeholder)
                @context[placeholder.name.to_sym].tap do |c|
                  raise "missing placeholder #{placeholder.name}" if c.nil?
                end
              else
                placeholder
              end

        return convert_to_opencl(var, shape_eval(var), data_type: placeholder.data_type, name: placeholder.name) unless var.is_a?(Tensor)
        Tensor.cast_dtype(var, placeholder.data_type)
      end

      def reduce_axis(current_axis, axis, val, keep_dims, f = ->(a, b) { a + b })
        return val unless val.is_a?(Array)

        r = val.collect do |v|
          reduce_axis(current_axis + 1, axis, v, keep_dims, f)
        end

        should_reduce_axis = axis.nil? || (axis.is_a?(Array) && axis.include?(current_axis)) || (current_axis == axis)

        if should_reduce_axis
          reduced_val = r[0]
          if r.size > 1
            reduced_val = f.call(r[0..val.size])
          elsif r.size.zero?
            reduced_val = f.call(nil)
          end
          keep_dims ? [ reduced_val ] : reduced_val
        else
          r
        end
      end

      # handle 3 tensor math operations
      def call_3way_vector_op(v_a, v_b, v_c, child_context, op = ->(a, b, c) { a + b + c })
        return op.call(v_a, v_b, v_c) unless v_a.is_a?(Array)

        v_a.each_with_index.collect do |v1, index|
          v2 = v_b[index]
          v3 = v_c[index]
          if v1.is_a?(Array)
            call_3way_vector_op(v1, v2, v3, child_context, op)
          else
            op.call(v1, v2, v3)
          end
        end
      end

      def all_true?(arr)
        if arr.is_a?(Array) || arr.is_a?(NArray)
          arr.each do |a|
            return false unless all_true?(a)
          end
          return true
        end

        arr != 0
      end

      def generate_vector(shape, dtype: :float32, generator:)
        if shape.is_a?(Integer)
          Array.new(shape) do
            generator.call
          end
        elsif shape.size > 1
          Array.new(shape[0]) do
            generate_vector(shape[1..shape.size], generator: generator, dtype: dtype)
          end
        elsif shape.size == 1
          Array.new(shape[0]) do
            generator.call
          end
        elsif shape.size.zero?
          generator.call
        end
      end

      def _get_randomizer(tensor, seed)
        if tensor.graph.random_seed && seed
          Random.new(tensor.graph.random_seed ^ seed)
        elsif tensor.graph.random_seed
          @session.randomizer[tensor.graph.object_id] ||= Random.new(tensor.graph.random_seed)
          @session.randomizer[tensor.graph.object_id]
        elsif seed
          @session.randomizer[tensor.operation] ||= Random.new(seed)
          @session.randomizer[tensor.operation]
        else
          Random.new
        end
      end

      def dump_intermediates
        arr = []
        arr << "============== start ==================="
        @context[:compute_history].each_with_index do |history, index|
          arr << "------------------------------------"
          arr << history[:name]
          arr << "#{history[:type]} #{history[:shape]}"
          arr << history[:source]
          arr << history[:description]
          arr << ""
          arr << history[:value].to_json
          arr << "------------------------------------"
        end
        arr << "============== end ====================="
        str = arr.join("\n")
        File.write("/tmp/intermediates.txt", str)
      end
    end
  end
end

TensorStream::Evaluator.register_evaluator(TensorStream::Evaluator::OpenclEvaluator, "opencl", 1)