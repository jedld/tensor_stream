require 'tensor_stream/evaluator/operation_helpers/random_gaussian'
require 'tensor_stream/evaluator/operation_helpers/array_ops_helper'
require 'tensor_stream/evaluator/operation_helpers/math_helper'
require 'tensor_stream/evaluator/base_evaluator'
require 'tensor_stream/evaluator/ruby/math_ops'
require 'tensor_stream/evaluator/ruby/nn_ops'
require 'tensor_stream/evaluator/ruby/array_ops'
require 'tensor_stream/evaluator/ruby/random_ops'
require 'tensor_stream/evaluator/ruby/images_ops'
require 'tensor_stream/evaluator/ruby/check_ops'

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
      include TensorStream::MathOps
      include TensorStream::NNOps
      include TensorStream::ArrayOps
      include TensorStream::RandomOps
      include TensorStream::ImagesOps
      include TensorStream::CheckOps

      def run(tensor, execution_context)
        return tensor.map { |t| run(t, execution_context) } if tensor.is_a?(Array) && !tensor.empty? && tensor[0].is_a?(Tensor)

        tensor = tensor.call if tensor.is_a?(Proc)

        child_context = execution_context.dup
        res = if tensor.is_a?(Operation)
                eval_operation(tensor, child_context)
              elsif tensor.is_a?(Variable)
                eval_variable(tensor, child_context)
              elsif tensor.is_a?(Placeholder)
                resolve_placeholder(tensor, child_context)
              elsif tensor.is_a?(OutputGroup)
                tensor.outputs[0]
              else
                eval_tensor(tensor, child_context)
              end
        execution_context.deep_merge!(returns: child_context[:returns])
        res
      end

      def run_with_buffer(tensor, context, execution_context)
        @context = context
        @context[:_cache][:_cl_buffers] ||= {} if context[:_cache]
        result = run(tensor, execution_context)
        TensorStream::Buffer.new(data_type: tensor.data_type, buffer: result)
      end

      def complete_eval(tensor, context)
        Kernel.loop do
          old_tensor = tensor
          tensor = run(tensor, context)

          tensor = tensor.map { |t| complete_eval(t, context) } if tensor.is_a?(Array) && !tensor.empty? && tensor[0].is_a?(Tensor)

          break if old_tensor.equal?(tensor)
          break unless tensor.is_a?(Tensor)
        end

        tensor.is_a?(OutputGroup) ? tensor.outputs[0] : tensor
      end

      protected

      def prepare_input(tensor, context, options = {})
        return nil unless tensor
        tensor = resolve_placeholder(tensor)
        if options[:noop]
          tensor
        elsif options[:no_eval]
          run(tensor, context)
        else
          complete_eval(tensor, context)
        end
      end

      def eval_variable(tensor, child_context)
        value = tensor.read_value
        raise "variable #{tensor.name} not initalized" if value.nil?

        eval_tensor(value, child_context).tap do |val|
          child_context[:returns] ||= {}
          child_context[:returns][:vars] ||= []
          child_context[:returns][:vars] << { name: tensor.name, val: val }
        end
      end

      register_op(:no_op, no_eval: true) do |_context, _tensor, inputs|
        inputs
      end

      register_op(:const) do |_context, _tensor, inputs|
        inputs[0]
      end

      register_op(:cast) do |context, tensor, inputs|
        call_op(tensor, inputs[0], context, ->(t, _b) { Tensor.cast_dtype(t, tensor.data_type) })
      end

      register_op(:sign) do |context, tensor, inputs|
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

        call_op(tensor, inputs[0], context, func)
      end

      register_op(:logical_and) do |context, tensor, inputs|
        call_vector_op(tensor, :logical_and, inputs[0], inputs[1], context, ->(t, u) { t && u })
      end

      register_op(:equal) do |context, tensor, inputs|
        call_vector_op(tensor, :equal, inputs[0], inputs[1], context, ->(t, u) { t == u })
      end

      register_op(:not_equal) do |context, tensor, inputs|
        call_vector_op(tensor, :not_equal, inputs[0], inputs[1], context, ->(t, u) { t != u })
      end

      def merge_dynamic_stitch(merged, indexes, data)
        indexes.each_with_index do |ind, m|
          if ind.is_a?(Array)
            merge_dynamic_stitch(merged, ind, data[m])
          else
            merged[ind] = data[m]
          end
        end
      end

      register_op :stop_gradient, no_eval: true do |_context, _tensor, inputs|
        inputs[0]
      end

      register_op :assign, noop: true do |context, tensor, _inputs|
        assign = tensor.inputs[0] || tensor
        assign.value = global_eval(tensor, tensor.inputs[1], context)
        assign.value
      end

      register_op :assign_add, noop: true do |context, tensor, _inputs|
        tensor.inputs[0].value = process_vector_math_op(tensor, tensor.inputs[0], tensor.inputs[1], context, ->(t, u) { t + u })
        tensor.inputs[0].value
      end

      register_op :variable, noop: true do |_context, tensor, _inputs|
        tensor.inputs[0].value
      end

      register_op :assign_sub, noop: true do |context, tensor, _inputs|
        tensor.inputs[0].value = process_vector_math_op(tensor, tensor.inputs[0], tensor.inputs[1], context, ->(t, u) { t - u })
        tensor.inputs[0].value
      end

      register_op :transpose do |_context, _tensor, inputs|
        shape = shape_eval(inputs[0])
        rank = get_rank(inputs[0])
        perm = inputs[1] || (0...rank).to_a.reverse
        if rank == 2 && perm.nil? # use native transpose for general case
          inputs[0].transpose
        else
          arr = inputs[0].flatten

          new_shape = perm.map { |p| shape[p] }
          new_arr = Array.new(shape.reduce(:*)) { 0 }
          transpose_with_perm(arr, new_arr, shape, new_shape, perm)
          TensorShape.reshape(new_arr, new_shape)
        end
      end

      register_op :less do |context, tensor, inputs|
        a, b = inputs
        call_vector_op(tensor, :less, a, b, context, ->(t, u) { t < u })
      end

      register_op :greater do |context, tensor, inputs|
        a, b = inputs
        call_vector_op(tensor, :greater, a, b, context, ->(t, u) { t > u })
      end

      register_op :greater_equal do |context, tensor, inputs|
        a, b = inputs
        call_vector_op(tensor, :greater_equal, a, b, context, ->(t, u) { t >= u })
      end

      register_op :less_equal do |context, tensor, inputs|
        a, b = inputs
        call_vector_op(tensor, :greater_equal, a, b, context, ->(t, u) { t <= u })
      end

      register_op :broadcast_transform do |_context, _tensor, inputs|
        broadcast(inputs[0], inputs[1])
      end

      register_op :identity do |_context, _tensor, inputs|
        inputs[0]
      end

      register_op :print do |_context, tensor, inputs|
        puts "#{tensor.options.fetch(:message, '')} #{inputs[1]}"
        inputs[0]
      end

      register_op %i[div real_div], noop: true do |context, tensor, inputs|
        process_vector_math_op(tensor, inputs[0], inputs[1], context, ->(t, u) { t / u })
      end

      register_op :broadcast_gradient_args do |_context, tensor, inputs|
        rx, ry = get_broadcast_gradient_args(inputs[0], inputs[1])
        OutputGroup.new([rx, ry], tensor.inputs.map(&:data_type))
      end

      register_op :flow_group, noop: true do |context, tensor, inputs|
        inputs.each { |input| global_eval(tensor, input, context) }
        nil # no output
      end

      register_op :softmax do |_context, _tensor, inputs|
        softmax(inputs[0])
      end

      register_op :save_ts do |_context, tensor, inputs|
        outputfile = inputs[0]
        inputs = tensor.inputs.dup

        basename = File.basename(outputfile)
        path = File.dirname(outputfile)

        new_filename = File.join(path, [basename, gs].compact.join('-'))

        inputs.shift
        variables = {}
        inputs.each do |savable|
          variables[savable.name] = TensorStream::Packer.pack(savable.read_value, savable.data_type)
        end
        File.write(new_filename, variables.to_yaml)
      end

      register_op :restore_v2 do |context, tensor, inputs|
        # prefix, tensor_names, shape_and_slices = inputs[0..3]
      end

      register_op :check_numerics do |context, tensor, inputs|
        message = tensor.options[:message]
        f = lambda { |t, _b|
          raise TensorStream::InvalidArgumentError, "#{message} Invalid argument" if t.nan? || t.infinite?
          t
        }
        call_op(tensor, inputs[0], context, f)
      end

      def eval_operation(tensor, child_context)
        return @context[tensor.name] if @context.key?(tensor.name)

        # puts "ruby eval #{object_id}: #{tensor.name}"
        invoke(tensor, child_context).tap do |result|
          # puts "result done ruby #{object_id}: #{tensor.name}"
          # assertions to make sure inferred shapes == actual evaluated shapes
          if tensor.shape.known? && (result.is_a?(Array) || result.is_a?(Float) || result.is_a?(Integer))
            if shape_eval(result) != tensor.shape.shape

              raise "assert error #{tensor.name} #{shape_eval(result)} != #{tensor.shape.shape}"
            end
          end

          if tensor.breakpoint
            a = resolve_placeholder(tensor.inputs[0], child_context) if tensor.inputs && tensor.inputs[0]
            b = resolve_placeholder(tensor.inputs[1], child_context) if tensor.inputs && tensor.inputs[1]
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
        raise e, "error #{e.message} while evaluating #{tensor.name}  defined at #{tensor.source}"
      rescue TensorStreamError => e
        raise e, "error #{e.message} while evaluating #{tensor.name}  defined at #{tensor.source}"
      rescue StandardError => e
        # a = resolve_placeholder(tensor.inputs[0], child_context) if tensor.inputs && tensor.inputs[0]
        # b = resolve_placeholder(tensor.inputs[1], child_context) if tensor.inputs && tensor.inputs[1]
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
        raise EvaluatorExcecutionException.new(e, tensor), "error #{e.message} while evaluating #{tensor.name} : #{tensor.to_math(true, 1)} defined at #{tensor.source}"
      end

      def eval_tensor(tensor, child_context)
        return tensor unless tensor.is_a?(Tensor)

        cache_key = "#{tensor.graph.object_id}_ruby_#{tensor.name}"
        return @context[cache_key] if @context.key?(cache_key)
        return @context[:_cache][cache_key] if @context[:_cache] && @context[:_cache].key?(tensor.name)

        if tensor.value.is_a?(Array)
          tensor.value.collect do |input|
            input.is_a?(Tensor) ? run(input, child_context) : input
          end
        else
          tensor.value.is_a?(Tensor) ? run(tensor.value, child_context) : tensor.value
        end.tap do |result|
          @context[cache_key] = result
          @context[:_cache][cache_key] = result if @context[:_cache] && tensor.is_const
        end
      end

      def convert_from_buffer(_tensor, result)
        result.buffer
      end

      private

      def get_op_with_axis(a, target_axis, current_axis, op)
        rank = get_rank(a)
        return a.send(:"#{op}_index") if rank == 1

        if current_axis == target_axis
          compare_items = a.collect(&:flatten).transpose
          compare_items.map { |item| item.index(item.send(:"#{op}")) }
        elsif a[0].is_a?(Array)
          a.map { |item| get_op_with_axis(item, target_axis, current_axis + 1, op) }
        else
          return a.send(:"#{op}_index")
        end
      end

      def reduction(child_context, tensor, func)
        val = global_eval(tensor, tensor.inputs[0], child_context)
        axis = global_eval(tensor, tensor.inputs[1], child_context)
        keep_dims = global_eval(tensor, tensor.options[:keepdims], child_context)
        reduce(val, axis, keep_dims, func)
      end

      def call_op(op, a, child_context, func)
        a = complete_eval(a, child_context)
        process_function_op(a, func)
      end

      def call_vector_op(tensor, op, a, b, child_context, func)
        process_vector_math_op(tensor, a, b, child_context, func)
      rescue FullEvalNotPossible
        TensorStream.send(op.to_sym, a, b)
      end

      def process_vector_math_op(tensor, a, b,  child_context, op)
        eval_a = global_eval(tensor, a, child_context) unless a.nil?
        eval_b = global_eval(tensor, b, child_context) unless b.nil?

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

      # multi array ops on ruby arrays with same sizes
      def multi_array_op(func, *args)
        elem = args[0]
        if (elem.is_a?(Array))
          elem.each_with_index.collect do |item, index|
            indexed_args = args.collect { |a| a[index] }
            multi_array_op(func, *indexed_args)
          end
        else
          func.call(*args)
        end
      end

      def _rank_from_shape(shape)
        shape.is_a?(Array) ? shape.size : 0
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
                  raise TensorStream::ValueError, "missing placeholder #{placeholder.name}" if c.nil?
                  if placeholder.shape.shape
                    value_shape = shape_eval(c)
                    placeholder_shape = placeholder.shape.shape
                    placeholder_shape.zip(value_shape).each do |p_shape, v_shape|
                      next if p_shape.nil?
                      raise TensorStream::ValueError, "placeholder expects #{placeholder_shape}, got #{value_shape}" if p_shape != v_shape
                    end
                  end
                end
              else
                placeholder
              end

        return var unless placeholder.is_a?(Tensor)
        Tensor.cast_dtype(var, placeholder.data_type)
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
        @context[:compute_history].each_with_index do |history, _index|
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
        File.write('/tmp/intermediates.txt', str)
      end
    end
  end
end

TensorStream::Evaluator.register_evaluator(TensorStream::Evaluator::RubyEvaluator, 'ruby')
