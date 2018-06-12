require 'tensor_stream/evaluator/operation_helpers/random_gaussian'
require 'tensor_stream/evaluator/operation_helpers/array_ops_helper'
require 'tensor_stream/evaluator/operation_helpers/math_helper'
require 'tensor_stream/evaluator/opencl_buffer'
require 'tensor_stream/evaluator/opencl_template_helper'
require 'distribution'
require 'opencl_ruby_ffi'
require 'narray_ffi'

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
    class OpenclEvaluator
      attr_accessor :retain

      include TensorStream::OpHelper
      include TensorStream::ArrayOpsHelper
      include TensorStream::MathHelper

      def initialize(session, context, thread_pool: nil, log_intermediates: false, preferred_device: nil)
        @session = session
        @context = context
        @log_intermediates = log_intermediates
        @preferred_device = preferred_device
        @retain = context[:retain] || []
        @thread_pool = thread_pool || Concurrent::ImmediateExecutor.new
        @context[:_cache][:_cl_buffers] ||= {}
        @context[:compute_history] = [] if log_intermediates
      end

      # opencl evaluator main entrypoint
      def run(tensor, execution_context)
        _create_opencl_context

        read_final_result(complete_eval(tensor, execution_context))

      end

      def complete_eval(tensor, context)
        create_command_queue
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
            device
          end
        end
        @context[:_cache][:_opencl_context] ||= OpenCL.create_context(opencl_device)
      end

      def choose_best_device
        @best_device ||= begin
          devices = OpenCL.platforms.flat_map do |p|
            p.devices.select { |d| d.available > 0 }.each_with_index.collect do |d, index|
              score = 0
              if d.type.to_s == 'CPU'
                score += 1
              elsif d.type.to_s == 'GPU'
                score += 4
              end

              score += d.max_compute_units

              [d, score, p.name, index]
            end
          end
        end
        devices.max { |a| a[1] }
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
          File.write("/tmp/#{kernel}.#{suffix}.cl", source)
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

        return tensor if retain.include?(tensor) # if var is in retain don't eval to value

        tensor = tensor.call if tensor.is_a?(Proc)

        child_context = execution_context.dup
        res = if tensor.is_a?(Operation)
                eval_operation(tensor, child_context)
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

      def eval_operation(tensor, child_context)
        return @context[tensor.name] if @context.key?(tensor.name)
        cache_key = "#{tensor.graph.object_id}_opencl_#{tensor.name}"
        return @context[cache_key] if @context.key?(cache_key)
        a = resolve_placeholder(tensor.items[0], child_context) if tensor.items && tensor.items[0]
        b = resolve_placeholder(tensor.items[1], child_context) if tensor.items && tensor.items[1]
        # puts tensor.name
        case tensor.operation
        when :concat
          input_a = read_final_result(complete_eval(a, child_context))
          arr = concat_array(input_a, tensor.options[:axis])
          convert_to_opencl(arr.flatten, shape_eval(arr), data_type: tensor.data_type, name: tensor.name)
        when :cond
          pred = complete_eval(tensor.options[:pred], child_context)
          a = _run(a, child_context)
          b = _run(b, child_context)

          if all_true?(pred.buffer)
            a
          else
            b
          end
        when :identity
          _run(a, child_context)
        when :eye
          rows = complete_eval(a, child_context)
          columns = complete_eval(b, child_context)
          shape = [rows.buffer[0], columns.buffer[0]]
          eye_arr = Array.new(rows.buffer[0]) do |i|
            Array.new(columns.buffer[0]) do |col|
              if fp_type?(tensor.data_type)
                i == col ? 1.0 : 0.0
              else
                i == col ? 1 : 0
              end
            end
          end

          convert_to_opencl(eye_arr.flatten, shape, data_type: tensor.data_type, name: tensor.name)
        when :pad
          a = read_final_result(complete_eval(a, child_context))
          p = read_final_result(complete_eval(tensor.options[:paddings], child_context))

          padding = arr_pad(a, p, tensor.data_type)
          convert_to_opencl(padding.flatten, shape_eval(padding), data_type: tensor.data_type, name: tensor.name)
        when :tile
          input = read_final_result(complete_eval(a, child_context))
          multiples = read_final_result(complete_eval(b, child_context))

          rank = get_rank(input)
          raise '1D or higher tensor required' if rank.zero?
          raise "invalid multiple size passed #{rank} != #{multiples.size}" if rank != multiples.size

          tile = tile_arr(input, 0, multiples)
          arr = tile.nil? ? [] : tile
          convert_to_opencl(arr.flatten, shape_eval(arr), data_type: tensor.data_type, name: tensor.name)
        when :assign
          assign_var(tensor, b, child_context)
        when :assign_add
          a = _run(a, child_context)
          b = _run(b, child_context)

          value = execute_2_operand_func('add', tensor, a, b, child_context)
          assign_var(tensor, value, child_context)
        when :assign_sub
          a = _run(a, child_context)
          b = _run(b, child_context)

          value = execute_2_operand_func('sub', tensor, a, b, child_context)
          assign_var(tensor, value, child_context)
        when :less
          execute_2_operand_func('less', tensor, a, b, child_context, 'cond')
        when :less_equal
          execute_2_operand_func('less_equal', tensor, a, b, child_context, 'cond')
        when :greater
          execute_2_operand_func('greater', tensor, a, b, child_context, 'cond')
        when :greater_equal
          execute_2_operand_func('greater_equal', tensor, a, b, child_context, 'cond')
        when :equal
          execute_2_operand_func('equal', tensor, a, b, child_context, 'cond')
        when :not_equal
          execute_2_operand_func('not_equal', tensor, a, b, child_context, 'cond')
        when :logical_and
          execute_2_operand_func('logical_and', tensor, a, b, child_context, 'cond')
        when :where
          pred = tensor.options[:pred]
          execute_cond_func('where', tensor, pred, a, b, child_context)
        when :max
          execute_2_operand_func('max', tensor, a, b, child_context)
        when :add
          execute_2_operand_func('add', tensor, a, b, child_context)
        when :div
          execute_2_operand_func('div', tensor, a, b, child_context)
        when :sub
          execute_2_operand_func('sub', tensor, a, b, child_context)
        when :matmul
          a = _run(a, child_context)
          b = _run(b, child_context)

          m = a.shape[0]
          n = b.shape[1]
          v = b.shape[0]
          k = a.shape[1]

          m, k = [a.shape[1], a.shape[0]] if tensor.options[:transpose_a]
          n, v = [b.shape[0], b.shape[1]] if tensor.options[:transpose_b]

          result_shape = [m, n]

          raise "#{tensor.items[0].name} rank must be greater than 1" if a.shape.size < 2
          raise "#{tensor.items[1].name} rank must be greater than 1" if b.shape.size < 2
          raise "incompatible shape sizes for matrix multiplication (#{a.shape[1]} != #{b.shape[0]}) #{a.shape} vs #{b.shape}" if k != v

          dtype = tensor.data_type
          a, b = type_cast(a, b, name: "#{tensor.name}/cast_#{a.name}_#{b.data_type}")
          output_buffer = _create_result_buffer(a.data_type, result_shape, tensor.name)

          cl_m = OpenCL::Int1.new(m)
          cl_n = OpenCL::Int1.new(n)
          cl_k = OpenCL::Int1.new(k)

          transpose_a = OpenCL::Int1.new(tensor.options[:transpose_a] ? 1 : 0)
          transpose_b = OpenCL::Int1.new(tensor.options[:transpose_b] ? 1 : 0)

          output_buffer.op = _cl_program('gemm').send(:"gemm_#{dtype}", _opencl_queue, result_shape, cl_m, cl_n, cl_k, transpose_a, transpose_b, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer)
          output_buffer
        when :mul
          execute_2_operand_func('mul', tensor, a, b, child_context)
        when :pow
          execute_2_operand_func('pow', tensor, a, b, child_context)
        when :cast
          a = _run(a, child_context)
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
        when :sign
          execute_func('sign', tensor, a, child_context)
        when :exp
          execute_func('exp', tensor, a, child_context)
        when :log
          execute_func('log', tensor, a, child_context)
        when :sin
          execute_func('sin', tensor, a, child_context)
        when :tan
          execute_func('tan', tensor, a, child_context)
        when :cos
          execute_func('cos', tensor, a, child_context)
        when :abs
          execute_func('abs', tensor, a, child_context)
        when :sqrt
          execute_func('sqrt', tensor, a, child_context)
        when :negate
          execute_func('negate', tensor, a, child_context)
        when :square
          execute_func('square', tensor, a, child_context)
        when :reciprocal
          execute_func('reciprocal', tensor, a, child_context)
        when :tanh
          execute_func('tanh', tensor, a, child_context)
        when :tanh_grad
          execute_func('tanh_grad', tensor, a, child_context)
        when :sigmoid
          execute_func('sigmoid', tensor, a, child_context)
        when :log1p
          execute_func('log1p', tensor, a, child_context)
        when :round
          execute_func('round', tensor, a, child_context)
        when :softmax
          a = _run(a, child_context)
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
        when :softmax_grad
          a = _run(a, child_context)
          grad = _run(b, child_context)
          event_wait_list = [a.op].compact
          dtype = tensor.data_type
          output_buffer = _create_result_buffer(tensor.data_type, a.shape, tensor.name)

          m, n = a.shape
          work_group = [m]
          n = m if n.nil?
          cl_n = OpenCL::Int1.new(n || 1)
          event = _cl_program("softmax_grad", dtype: dtype, size: n).send(:"softmax_grad_#{dtype}", _opencl_queue, work_group, cl_n, a.cl_buffer, grad.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
          output_buffer.op = event
          output_buffer
        when :sigmoid_grad
          execute_2_operand_func('sigmoid_grad', tensor, a, b, child_context)
        when :truncate
          a = _run(a, child_context)
          b = _run(b, child_context)

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
        when :check_numerics
          a = complete_eval(a, child_context)
          name = tensor.options[:name]

          a.buffer.each do |item|
            raise "#{name} Invalid Argument" if item.nan? || item.infinite?
          end
          a
        when :zeros, :ones, :zeros_like, :ones_like
          shape = if %i[zeros_like ones_like].include?(tensor.operation)
            _run(a, child_context).shape
          else
            read_final_result(complete_eval(a, child_context)) || tensor.shape.shape
          end

          func = if %i[zeros zeros_like].include?(tensor.operation)
            -> { tensor.data_type == :int32 ? 0 : 0.0 }
          else
            -> { tensor.data_type == :int32 ? 1 : 1.0 }
          end

          size = shape.empty? ? 1 : shape.reduce(:*)

          buffer = if TensorStream::Ops::FLOATING_POINT_TYPES.include?(tensor.data_type)
                      NArray.sfloat(size)
                    elsif TensorStream::Ops::INTEGER_TYPES.include?(tensor.data_type)
                      NArray.int(size)
                    else
                      raise "unsupported type #{tensor.data_type}"
                    end

          data = if !shape.empty?
            Array.new(size) do |index|
              func.call
            end
          else
            func.call
          end

          convert_to_opencl(data, shape, data_type: tensor.data_type, name: tensor.name)
         when :broadcast_transform
          a = _run(a, child_context)
          b = _run(b, child_context)

         if a.shape == b.shape
           [a, b]
         else
           input_a = read_final_result(complete_eval(a, child_context))
           input_b = read_final_result(complete_eval(b, child_context))
           b_a, b_b = broadcast(input_a, input_b)
           [ wrap_opencl(b_a, data_type: a.data_type, name: "#{tensor.name}_a"),
             wrap_opencl(b_b, data_type: a.data_type, name: "#{tensor.name}_b")]
         end
        when :print
          a = _run(a, child_context)
          b = _run(b, child_context)
          input_b = complete_eval(b, child_context)
          input_b = read_final_result(input_b)
          puts "#{tensor.options.fetch(:message, '')} #{input_b}"
          a
        when :rank
          a = _run(a, child_context)
          wrap_opencl(a.shape.size, data_type: tensor.data_type, name: tensor.name)
        when :stop_gradient
          _run(a, child_context)
        when :slice
          input_a = complete_eval(a, child_context)
          input_b = read_final_result(complete_eval(b, child_context))
          size = tensor.options[:size]

          slice_param = input_b.zip(size).collect { |x, y| x..x + y - 1 }.reverse

          new_buf = input_a.buffer.reshape(*input_a.shape.reverse)
          sliced = new_buf.slice[*slice_param]
          convert_to_opencl(sliced.flatten, sliced.shape.reverse, data_type: a.data_type, name: tensor.name)
        when :transpose
          input_a = complete_eval(a, child_context)
          t_param = Array.new(input_a.shape.size) { |index| index }.reverse
          transposed = input_a.buffer.reshape(*input_a.shape.reverse).transpose(*t_param)
          convert_to_opencl(transposed.flatten, transposed.shape.reverse, data_type: a.data_type, name: tensor.name)
        when :index
          a = complete_eval(a, child_context)
          input_a = read_final_result(a)
          index = read_final_result(complete_eval(b, child_context))

          if a.is_a?(Array)
            a[index]
          else
            new_shape = a.shape.dup
            new_shape.shift
            convert_to_opencl(input_a[index], new_shape, data_type: a.data_type, name: tensor.name)
          end
        when :broadcast_gradient_args
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)

          wrap_opencl(get_broadcast_gradient_args(a.buffer.to_a, b.buffer.to_a), data_type: a.data_type, name: tensor.name)
        when :shape
          a = _run(a, child_context)

          wrap_opencl(a.shape, name: tensor.name, data_type: tensor.options[:out_type] || :float32)
        when :reshape
          arr = complete_eval(a, child_context)
          new_shape = read_final_result(complete_eval(b, child_context))

          if new_shape.size.zero? && arr.buffer.size == 1
            arr.shape = new_shape
            arr
          else
            new_shape = TensorShape.fix_inferred_elements(new_shape, arr.buffer.size)
            arr.shape = new_shape
            arr
          end
        when :random_uniform
          maxval = tensor.options.fetch(:maxval, 1)
          minval = tensor.options.fetch(:minval, 0)
          seed = tensor.options[:seed]

          random = _get_randomizer(tensor, seed)
          generator = -> { random.rand * (maxval - minval) + minval }
          shape = tensor.options[:shape] || tensor.shape.shape

          convert_to_opencl(generate_vector(shape, generator: generator), shape, data_type: tensor.data_type, name: tensor.name)
        when :random_normal
          random = _get_randomizer(tensor, seed)
          r = RandomGaussian.new(tensor.options.fetch(:mean), tensor.options.fetch(:stddev), -> { random.rand })
          random = _get_randomizer(tensor, seed)
          generator = -> { r.rand }
          shape = tensor.options[:shape] || tensor.shape.shape

          convert_to_opencl(generate_vector(shape, generator: generator), shape, data_type: tensor.data_type, name: tensor.name)
        when :glorot_uniform
          random = _get_randomizer(tensor, seed)

          shape = tensor.options[:shape] || tensor.shape.shape
          fan_in, fan_out = if shape.size.zero?
                              [1, 1]
                            elsif shape.size == 1
                              [1, shape[0]]
                            else
                              [shape[0], shape.last]
                            end

          limit = Math.sqrt(6.0 / (fan_in + fan_out))

          minval = -limit
          maxval = limit

          generator = -> { random.rand * (maxval - minval) + minval }
          convert_to_opencl(generate_vector(shape, generator: generator), shape, data_type: tensor.data_type, name: tensor.name)
        when :flow_group
          tensor.items.collect { |item| _run(item, child_context) }
        when :sum
          reduction(child_context, tensor, a, b, :sum)
        when :mean
          reduction(child_context, tensor, a, b, :mean)
        when :prod
          input_a = complete_eval(a, child_context)
          if input_a.buffer.empty?
            convert_to_opencl([1.0], [], data_type: a.data_type, name: tensor.name)
          else
            reduction(child_context, tensor, a, b, :prod)
          end
        when :argmin
          a = complete_eval(a, child_context)
          axis = tensor.options[:axis] || 0
          arr = a.buffer.reshape(*a.shape.reverse).to_a
          op = get_op_with_axis(arr, axis, 0, a.data_type, ->(a, b) { a < b })
          convert_to_opencl(op, shape_eval(op), data_type: tensor.data_type, name: tensor.name)
        when :argmax
          a = complete_eval(a, child_context)
          axis = tensor.options[:axis] || 0
          arr = a.buffer.reshape(*a.shape.reverse).to_a
          op = get_op_with_axis(arr, axis, 0, a.data_type, ->(a, b) { a > b })
          convert_to_opencl(op, shape_eval(op), data_type: tensor.data_type, name: tensor.name)
        else
          raise "unknown op #{tensor.operation}"
        end.tap do |result|
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
        binding.pry
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
        assign = tensor.items[0] || tensor
        buffer = complete_eval(b, child_context)
        if assign.buffer
          assign.buffer.op = _opencl_queue.enqueue_write_buffer(assign.buffer.cl_buffer, buffer.buffer)
        else
          assign.buffer = convert_to_opencl(read_final_result(buffer), buffer.shape, data_type: tensor.data_type, name: tensor.name)
        end
        assign.buffer.dirty = true
        assign.buffer
      end

      def execute_2_operand_func(op_name, tensor, input_a, input_b, child_context, prog_name = nil)
        a = _run(input_a, child_context)
        b = _run(input_b, child_context)
        a, b = type_cast(a, b, name: "#{tensor.name}/cast_#{a.name}_#{b.data_type}")
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

        event = if prog == "#{op_name}_b"
          cl_m_b, cl_n_b = if b.shape.size == 2
            [ OpenCL::Int1.new(b.shape[0]), OpenCL::Int1.new(b.shape[1]) ]
          elsif b.shape.size == 1
            [ OpenCL::Int1.new(1), OpenCL::Int1.new(b.shape[0]) ]
          else
            raise "rank > 2 not supported!"
          end
          _cl_program("#{prog_name || op_name}", dtype: dtype).send(:"#{prog}_#{dtype}", _opencl_queue, work_group, cl_m, cl_n, cl_m_b, cl_n_b, cl_switch, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
        else
          _cl_program("#{prog_name || op_name}", dtype: dtype).send(:"#{prog}_#{dtype}", _opencl_queue, work_group, cl_m, cl_n, cl_switch, a.cl_buffer, b.cl_buffer, output_buffer.cl_buffer, event_wait_list: event_wait_list)
        end

        output_buffer.op = event
        output_buffer
      end

      def execute_cond_func(op_name, tensor, pred, input_a, input_b, child_context)
        p = _run(pred, child_context)
        a = _run(input_a, child_context)
        b = _run(input_b, child_context)

        a, b = type_cast(a, b, name: "#{tensor.name}/cast_#{a.name}_#{b.data_type}")
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

      def type_cast(a, b, name: nil)
        return [a, b] if a.data_type == b.data_type
        m, n = b.shape
        work_group = [m || 1, n || 1]
        buffer = _create_result_buffer(b.data_type, b.shape, name)
        if (TensorStream::Ops::FLOATING_POINT_TYPES.include?(a.data_type.to_sym))
          if TensorStream::Ops::INTEGER_TYPES.include?(b.data_type.to_sym)
            cl_m = OpenCL::Int1.new(m || 1)
            cl_n = OpenCL::Int1.new(n || 1)

            _cl_program("cast", source_dt: a.data_type, target_dt: b.data_type).cast(_opencl_queue, work_group, cl_m, cl_n, b.cl_buffer, buffer.cl_buffer)
            return [a, buffer]
          end
        elsif TensorStream::Ops::INTEGER_TYPES.include?(a.data_type.to_sym)
          if TensorStream::Ops::FLOATING_POINT_TYPES.include?(b.data_type.to_sym)
            cl_m = OpenCL::Int1.new(m || 1)
            cl_n = OpenCL::Int1.new(n || 1)
            _cl_program("cast", source_dt: a.data_type, target_dt: b.data_type).cast(_opencl_queue, work_group, cl_m, cl_n, b.cl_buffer, buffer.cl_buffer)
            return [a, buffer]
          end
        end

        [a, b]
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

        cache_key = "_cl_object_#{name}_#{shape.join('_')}"
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
              cl_object.buffer[index] = Tensor.cast_dtype(element, data_type)
            end
          end
        elsif value.is_a?(NArray)
          cl_object.buffer = value
        else
          cl_object.buffer[0] = Tensor.cast_dtype(value, data_type)
        end

        write_op = if cl_object.cl_buffer && !value.nil? && (!value.is_a?(Array) || !value.empty?)
          _opencl_queue.enqueue_write_buffer(cl_object.cl_buffer, cl_object.buffer)
        end
        cl_object.op = write_op
        cl_object
      end

      def allocate_narray_for_type(data_type, narray_size)
        if TensorStream::Ops::FLOATING_POINT_TYPES.include?(data_type.to_sym) || TensorStream::Ops::FLOATING_POINT_TYPES.include?(data_type.to_sym)
          NArray.sfloat(narray_size)
        elsif TensorStream::Ops::INTEGER_TYPES.include?(data_type.to_sym) || TensorStream::Ops::INTEGER_TYPES.include?(data_type.to_sym)
          NArray.int(narray_size)
        elsif data_type.to_sym == :boolean
          NArray.int(narray_size)
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
        return placeholder if retain.include?(placeholder)

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
          elsif r.size == 0
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
