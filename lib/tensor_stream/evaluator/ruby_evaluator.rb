require 'tensor_stream/evaluator/operation_helpers/random_gaussian'
require 'tensor_stream/evaluator/operation_helpers/array_ops_helper'
require 'tensor_stream/evaluator/operation_helpers/math_helper'
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
    class RubyEvaluator < BaseEvaluator
      attr_accessor :retain

      include TensorStream::OpHelper
      include TensorStream::ArrayOpsHelper
      include TensorStream::MathHelper

      def initialize(session, context, thread_pool: nil, log_intermediates: false)
        @session = session
        @context = context
        @log_intermediates = log_intermediates
        @retain = context[:retain] || []
        @thread_pool = thread_pool || Concurrent::ImmediateExecutor.new

        @context[:compute_history] = [] if log_intermediates
      end

      def run(tensor, execution_context)
        if tensor.is_a?(Array) && tensor.size > 0 && tensor[0].is_a?(Tensor)
          return tensor.map { |t| run(t, execution_context) }
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

      def complete_eval(tensor, context)
        Kernel.loop do
          old_tensor = tensor
          tensor = run(tensor, context)

          tensor = tensor.map { |t| complete_eval(t, context) } if tensor.is_a?(Array) && !tensor.empty? && tensor[0].is_a?(Tensor)

          return tensor if old_tensor.equal?(tensor)
          return tensor unless tensor.is_a?(Tensor)
        end
      end

      protected

      def eval_variable(tensor, child_context)
        value = tensor.read_value
        if value.nil?
          raise "variable #{tensor.name} not initalized"
        end
        eval_tensor(value, child_context).tap do |val|
          child_context[:returns] ||= {}
          child_context[:returns][:vars] ||= []
          child_context[:returns][:vars] << { name: tensor.name, val: val }
        end
      end

      def eval_operation(tensor, child_context)
        return @context[tensor.name] if @context.key?(tensor.name)
        a = resolve_placeholder(tensor.items[0], child_context) if tensor.items && tensor.items[0]
        b = resolve_placeholder(tensor.items[1], child_context) if tensor.items && tensor.items[1]
        # puts tensor.name
        case tensor.operation
        when :const
          complete_eval(a, child_context)
        when :argmax
          a = complete_eval(a, child_context)
          axis = tensor.options[:axis] || 0

          get_op_with_axis(a, axis, 0, tensor.data_type)
        when :argmin
          a = complete_eval(a, child_context)
          axis = tensor.options[:axis] || 0

          get_op_with_axis(a, axis, 0, tensor.data_type, ->(a, b) { a < b })
        when :cast
          a = complete_eval(a, child_context)

          call_op(:cast, a, child_context, ->(t, _b) { Tensor.cast_dtype(t, tensor.data_type) })
        when :sign
          a = complete_eval(a, child_context)

          func = lambda { |x, _b|
            if x.zero? || (x.is_a?(Float) && x.nan?)
              0
            elsif x < 0
              -1
            elsif x > 0
              1
            else
              raise 'assert: cannot be here'
            end
          }

          call_op(:sign, a, child_context, func)
        when :logical_and
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)

          call_vector_op(:greater, a, b, child_context, ->(t, u) { t && u })
        when :equal
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)

          call_vector_op(:greater, a, b, child_context, ->(t, u) { t == u })
        when :not_equal
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)

          call_vector_op(:not_equal, a, b, child_context, ->(t, u) { t != u })
        when :index
          f = run(a, child_context)
          index = run(b, child_context)

          f[index]
        when :slice
          input = complete_eval(a, child_context)
          start = complete_eval(b, child_context)
          size = complete_eval(tensor.options[:size], child_context)
          raise "start index and size not of the same shape #{start.size} != #{size.size}" if start.size != size.size
          slice_tensor(input, start, size)
        when :negate
          call_vector_op(:negate, a, nil, child_context, ->(t, _u) { -t })
        when :add
          call_vector_op(:add, a, b, child_context, ->(t, u) { t + u })
        when :sub
          call_vector_op(:sub, a, b, child_context, ->(t, u) { t - u })
        when :mul
          call_vector_op(:mul, a, b, child_context, ->(t, u) { t * u })
        when :pow
          call_vector_op(:pow, a, b, child_context, ->(t, u) { t**u })
        when :concat
          values = complete_eval(a, child_context)
          concat_array(values, tensor.options[:axis])
        when :round
          call_op(:round, a, child_context, ->(t, _b) { t.round })
        when :abs
          call_op(:abs, a, child_context, ->(t, _b) { t.abs })
        when :tanh
          call_op(:tanh, a, child_context, ->(t, _b) { Math.tanh(t) })
        when :tan
          call_op(:tan, a, child_context, ->(t, _b) { Math.tan(t) })
        when :sec
          call_op(:sec, a, child_context, ->(t, _b) { Math.sec(t) })
        when :sin
          call_op(:sin, a, child_context, ->(t, _b) { Math.sin(t) })
        when :cos
          call_op(:cos, a, child_context, ->(t, _b) { Math.cos(t) })
        when :log1p
          call_op(:log1p, a, child_context, ->(t, _b) { Math.log(1 + t) })
        when :log
          call_op(:log, a, child_context, ->(t, _b) { t < 0 ? Float::NAN : Math.log(t) })
        when :exp
          call_op(:exp, a, child_context, ->(t, _b) { Math.exp(t) })
        when :sigmoid
          call_op(:sigmoid, a, child_context, ->(t, _b) { sigmoid(t) })
        when :sigmoid_grad
          call_vector_op(:sigmoid_grad, a, b, child_context, ->(t, u) { u * sigmoid(t) * (1 - sigmoid(t)) })
        when :sqrt
          call_op(:exp, a, child_context, ->(t, _b) { Math.sqrt(t) })
        when :square
          call_op(:square, a, child_context, ->(t, _b) { t * t })
        when :reciprocal
          call_op(:square, a, child_context, ->(t, _b) { 1 / t })
        when :stop_gradient
          run(a, child_context)
        when :random_uniform
          maxval = tensor.options.fetch(:maxval, 1)
          minval = tensor.options.fetch(:minval, 0)
          seed = tensor.options[:seed]

          random = _get_randomizer(tensor, seed)
          generator = -> { random.rand * (maxval - minval) + minval }
          shape = tensor.options[:shape] || tensor.shape.shape
          generate_vector(shape, generator: generator)
        when :random_normal
          random = _get_randomizer(tensor, seed)
          r = RandomGaussian.new(tensor.options.fetch(:mean), tensor.options.fetch(:stddev), -> { random.rand })
          random = _get_randomizer(tensor, seed)
          generator = -> { r.rand }
          shape = tensor.options[:shape] || tensor.shape.shape
          generate_vector(shape, generator: generator)
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
          generate_vector(shape, generator: generator)
        when :flow_group
          tensor.items.collect { |item| run(item, child_context) }
        when :assign
          assign = tensor.items[0] || tensor
          assign.value = complete_eval(tensor.items[1], child_context)
          assign.value
        when :assign_add
          tensor.items[0].value = process_vector_math_op(tensor.items[0], tensor.items[1], child_context, ->(t, u) { t + u })
          tensor.items[0].value
        when :assign_sub
          tensor.items[0].value = process_vector_math_op(tensor.items[0], tensor.items[1], child_context, ->(t, u) { t - u })
          tensor.items[0].value
        when :mean
          c = fp_type?(tensor.data_type) ? 0.0 : 0
          func = lambda do |arr|
            return c if arr.nil?

            reduced_val = arr[0]
            arr[1..arr.size].each do |v|
              reduced_val = vector_op(reduced_val, v, ->(a, b) { a + b })
            end

            vector_op(reduced_val, nil, ->(a, _b) { a / arr.size })
          end

          reduction(child_context, tensor, func)
        when :sum
          c = fp_type?(tensor.data_type) ? 0.0 : 0
          func = lambda do |arr|
            reduced_val = arr[0]
            arr[1..arr.size].each do |v|
              reduced_val = vector_op(reduced_val, v, ->(t, u) { t + u })
            end
            reduced_val
          end

          reduction(child_context, tensor, func)
        when :tanh_grad
          x = complete_eval(a, child_context)
          call_op(:tanh_grad, x, child_context, ->(t, _b) { 1 - Math.tanh(t) * Math.tanh(t) })
        when :prod
          c = fp_type?(tensor.data_type) ? 1.0 : 1
          func = lambda do |arr|
            return c if arr.nil?

            reduced_val = arr[0]
            arr[1..arr.size].each do |v|
              reduced_val = vector_op(reduced_val, v, ->(a, b) { a * b })
            end
            reduced_val
          end

          reduction(child_context, tensor, func)
        when :transpose
          matrix_a = complete_eval(a, child_context)
          matrix_a.transpose
        when :eye
          rows = complete_eval(a, child_context)
          columns = complete_eval(b, child_context)

          Array.new(rows) do |i|
            Array.new(columns) do |col|
              if fp_type?(tensor.data_type)
                i == col ? 1.0 : 0.0
              else
                i == col ? 1 : 0
              end
            end
          end
        when :cond
          pred = complete_eval(tensor.options[:pred], child_context)

          if all_true?(pred)
            complete_eval(a, child_context)
          else
            complete_eval(b, child_context)
          end
        when :where
          pred = complete_eval(tensor.options[:pred], child_context)
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)

          call_3way_vector_op(pred, a, b, child_context, ->(t, u, v) { t ? u : v })
        when :less
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)

          call_vector_op(:greater, a, b, child_context, ->(t, u) { t < u })
        when :greater
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)

          call_vector_op(:greater, a, b, child_context, ->(t, u) { t > u })
        when :greater_equal
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)

          call_vector_op(:greater_equal, a, b, child_context, ->(t, u) { t >= u })
        when :less_equal
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)

          call_vector_op(:less_equal, a, b, child_context, ->(t, u) { t <= u })
        when :zeros, :ones, :zeros_like, :ones_like

          shape = if %i[zeros_like ones_like].include?(tensor.operation)
                    a = complete_eval(a, child_context)
                    shape_eval(a)
                  else
                    complete_eval(a, child_context) || tensor.shape.shape
                  end

          func = if %i[zeros zeros_like].include?(tensor.operation)
                   -> { tensor.data_type == :int32 ? 0 : 0.0 }
                 else
                   -> { tensor.data_type == :int32 ? 1 : 1.0 }
                 end

          if shape.is_a?(Array) && shape.size.zero?
            func.call
          else
            shape = [shape.to_i] unless shape.is_a?(Array)

            cache_key = "#{tensor.operation}_#{shape.to_s}"
            if @context[:_cache].key?(cache_key)
              return @context[:_cache][cache_key]
            else
              generate_vector(shape, generator: func).tap do |v|
                @context[:_cache][cache_key] = v
              end
            end
          end
        when :shape
          input = complete_eval(a, child_context)
          shape_eval(input, tensor.options[:out_type])
        when :matmul
          matrix_a = complete_eval(a, child_context)
          matrix_b = complete_eval(b, child_context)

          rank_a = get_rank(matrix_a)
          rank_b = get_rank(matrix_b)

          raise "#{tensor.items[0].name} rank must be greater than 1" if rank_a < 2
          raise "#{tensor.items[1].name} rank must be greater than 1" if rank_b < 2

          matrix_a = matrix_a.transpose if tensor.options[:transpose_a]
          matrix_b = matrix_b.transpose if tensor.options[:transpose_b]

          # handle matrix multiplication with constants like 1 or 0
          matrix_a = matmul_const_transform(matrix_a, matrix_b, tensor)
          matrix_b = matmul_const_transform(matrix_b, matrix_a, tensor)

          # check matrix dimensions
          raise "incompatible shape sizes for matrix multiplication (#{matrix_a[0].size} != #{matrix_b.size}) #{shape_eval(matrix_a)} vs #{shape_eval(matrix_b)}" if matrix_a[0].size != matrix_b.size

          (Matrix[*matrix_a] * Matrix[*matrix_b]).to_a
        when :gradients
          raise 'not implemented in evaluator' # see TensorStream.gradients instead.
        when :broadcast_transform
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)
          broadcast(a, b)
        when :truncate
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)
          truncate(a, b)
        when :identity
          complete_eval(a, child_context)
        when :print
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)
          puts "#{tensor.options.fetch(:message, '')} #{b}"
          a
        when :rank
          a = complete_eval(a, child_context)
          get_rank(a)
        when :div
          process_vector_math_op(a, b, child_context, ->(t, u) { t / u })
        when :reshape
          arr = complete_eval(a, child_context)
          new_shape = complete_eval(b, child_context)

          arr = [arr] unless arr.is_a?(Array)

          flat_arr = arr.flatten
          return flat_arr[0] if new_shape.size.zero? && flat_arr.size == 1

          new_shape = TensorShape.fix_inferred_elements(new_shape, flat_arr.size)

          TensorShape.reshape(flat_arr, new_shape)
        when :pad
          a = complete_eval(a, child_context)
          p = complete_eval(tensor.options[:paddings], child_context)

          arr_pad(a, p, tensor.data_type)
        when :max
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)

          call_vector_op(:max, a, b, child_context, ->(t, u) { [t, u].max })
        when :broadcast_gradient_args
          a = complete_eval(a, child_context)
          b = complete_eval(b, child_context)

          get_broadcast_gradient_args(a, b)
        when :reduced_shape
          input_shape = complete_eval(a, child_context)
          axes = complete_eval(b, child_context)

          return [] if axes.nil? # reduce to scalar
          axes = [ axes ] unless axes.is_a?(Array)
          return input_shape if axes.empty?

          axes.each do |dimen|
            input_shape[dimen] = 1
          end
          input_shape
        when :tile
          input = complete_eval(a, child_context)
          multiples = complete_eval(b, child_context)

          rank = get_rank(input)
          raise '1D or higher tensor required' if rank.zero?
          raise "invalid multiple size passed #{rank} != #{multiples.size}" if rank != multiples.size

          tile = tile_arr(input, 0, multiples)
          tile.nil? ? [] : tile
        when :softmax
          input = complete_eval(a, child_context)
          softmax(input)
        when :softmax_grad
          input = complete_eval(a, child_context)
          grad = complete_eval(b, child_context)
          softmax_input = softmax(input)
          f_grad = softmax_grad(softmax_input)
          f_grad.transpose.each_with_index.collect do |row, index|
            sum = 0.0
            row.each_with_index do |r, g_index|
              sum += r * grad[g_index]
            end
            sum
          end
        when :check_numerics
          a = complete_eval(a, child_context)
          message = tensor.options[:message]
          f = ->(t, _b) { raise  "#{message} Invalid argument" if t.nan? || t.infinite?; t }
          call_op(:check_numerics, a, child_context, f)
        else
          raise "unknown op #{tensor.operation}"
        end.tap do |result|
          if tensor.breakpoint
            a = complete_eval(a, child_context)
            b = complete_eval(b, child_context)

            tensor.breakpoint.call(tensor, a, b, complete_eval(result, child_context))
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
          @context[tensor.name] = result
        end
      rescue EvaluatorExcecutionException => e
        raise e
      rescue StandardError => e
        puts e.message
        puts e.backtrace.join("\n")

        # shape_a = a.shape.shape if a
        # shape_b = b.shape.shape if b
        # dtype_a = a.data_type if a
        # dtype_b = b.data_type if b
        a = complete_eval(a, child_context)
        b = complete_eval(b, child_context)
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

        cache_key = "#{tensor.graph.object_id}_ruby_#{tensor.name}"
        return @context[cache_key] if @context.key?(cache_key)
        return @context[:_cache][cache_key] if @context[:_cache] && @context[:_cache].key?(tensor.name)

        if tensor.value.is_a?(Array)
          tensor.value.collect do |item|
            item.is_a?(Tensor) ? run(item, child_context) : item
          end
        else
          tensor.value.is_a?(Tensor) ? run(tensor.value, child_context) : tensor.value
        end.tap do |result|
          @context[cache_key] = result
          @context[:_cache][cache_key] = result if @context[:_cache] && tensor.is_const
        end
      end

      private

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

      def reduction(child_context, tensor, func)
        val = complete_eval(tensor.items[0], child_context)
        axis = complete_eval(tensor.items[1], child_context)
        keep_dims = complete_eval(tensor.options[:keepdims], child_context)
        rank = get_rank(val)
        return val if axis && axis.is_a?(Array) && axis.empty?

        axis = if axis.nil?
          nil
        elsif axis.is_a?(Array)
          return val if axis.empty?

          axis.map { |a| a < 0 ? rank - a.abs : a }
        else
          axis < 0 ? rank - axis.abs : axis
        end

        reduce_axis(0, axis, val, keep_dims, func)
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

      def call_op(op, a, child_context, func)
        a = complete_eval(a, child_context)
        process_function_op(a, func)
      rescue FullEvalNotPossible
        TensorStream.send(op.to_sym, a)
      end

      def call_vector_op(op, a, b, child_context, func)
        process_vector_math_op(a, b, child_context, func)
      rescue FullEvalNotPossible
        TensorStream.send(op.to_sym, a, b)
      end

      def process_vector_math_op(a, b,  child_context, op)
        eval_a = complete_eval(a, child_context) unless a.nil?
        eval_b = complete_eval(b, child_context) unless b.nil?

        raise FullEvalNotPossible.new, "full eval not possible for #{a.name}" if eval_a.is_a?(Tensor) || eval_b.is_a?(Tensor)

        # ruby scalar
        eval_a, eval_b = broadcast(eval_a, eval_b)
        vector_op(eval_a, eval_b, op)
        # if get_rank(eval_a).zero?
        #   if get_rank(eval_b).zero?
        #     op.call(eval_a, eval_b)
        #   else
        #     vector_op(eval_b, eval_a, op, true)
        #   end
        # else
        #   vector_op(eval_a, eval_b, op)
        # end
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

        return var unless placeholder.is_a?(Tensor)
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
          v3 = v_c.is_a?(Array) ? v_c[index] : v_c
          if v1.is_a?(Array)
            call_3way_vector_op(v1, v2, v3, child_context, op)
          else
            op.call(v1, v2, v3)
          end
        end
      end

      def all_true?(arr)
        if arr.is_a?(Array)
          arr.each do |a|
            return false unless all_true?(a)
          end
          return true
        end

        !!arr
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

TensorStream::Evaluator.register_evaluator(TensorStream::Evaluator::RubyEvaluator, "ruby")