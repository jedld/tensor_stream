require "tensor_stream/evaluator/operation_helpers/random_gaussian"
require "tensor_stream/evaluator/operation_helpers/array_ops_helper"
require "tensor_stream/evaluator/operation_helpers/math_helper"
require "tensor_stream/evaluator/base_evaluator"
require "tensor_stream/evaluator/ruby/math_ops"
require "tensor_stream/evaluator/ruby/nn_ops"
require "tensor_stream/evaluator/ruby/array_ops"
require "tensor_stream/evaluator/ruby/random_ops"
require "tensor_stream/evaluator/ruby/images_ops"
require "tensor_stream/evaluator/ruby/check_ops"

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
        elsif !tensor.is_a?(Tensor)
          tensor
        else
          tensor.op
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

        tensor.is_a?(OutputGroup) ? tensor.outputs : tensor
      end

      protected

      def prepare_input(tensor, context, options = {})
        return nil unless tensor

        if options[:noop]
          tensor
        elsif options[:no_eval]
          run(tensor, context)
        else
          complete_eval(tensor, context)
        end
      end

      register_op(:no_op, no_eval: true) do |_context, _tensor, inputs|
        inputs
      end

      register_op(:const) do |_context, tensor, _inputs|
        tensor.options[:value]
      end

      register_op(:cast) do |context, tensor, inputs|
        call_op(inputs[0], context) { |t, _b| Tensor.cast_dtype(t, tensor.data_type) }
      end

      register_op(:sign) do |context, _tensor, inputs|
        call_op(inputs[0], context) do |x, _b|
          if x.zero? || (x.is_a?(Float) && x.nan?)
            0
          elsif x < 0
            -1
          elsif x > 0
            1
          else
            raise "assert: cannot be here"
          end
        end
      end

      register_op(:logical_and) do |context, tensor, inputs|
        call_vector_op(tensor, :logical_and, inputs[0], inputs[1], context) { |t, u| t && u }
      end

      register_op(:equal) do |context, tensor, inputs|
        call_vector_op(tensor, :equal, inputs[0], inputs[1], context) { |t, u| t == u }
      end

      register_op(:not_equal) do |context, tensor, inputs|
        call_vector_op(tensor, :not_equal, inputs[0], inputs[1], context) { |t, u| t != u }
      end

      register_op :placeholder, no_eval: true do |context, tensor, _inputs|
        ph = @context[tensor.name.to_sym].tap do |c|
          raise TensorStream::ValueError, "missing placeholder #{tensor.name}" if c.nil?

          if tensor.shape.shape
            value_shape = shape_eval(c)
            placeholder_shape = tensor.shape.shape
            placeholder_shape.zip(value_shape).each do |p_shape, v_shape|
              next if p_shape.nil?
              raise TensorStream::ValueError, "placeholder expects #{placeholder_shape}, got #{value_shape}" if p_shape != v_shape
            end
          end
        end
        if ph.is_a?(Tensor)
          raise TensorStream::ValueError, "placeholder expects type #{tensor.data_type}, got #{ph.data_type}" if ph.data_type != tensor.data_type

          global_eval(tensor, ph, context)
        else
          global_eval(tensor, Tensor.cast_dtype(ph, dtype: tensor.data_type), context)
        end
      end

      register_op :variable_v2, no_eval: true do |_context, tensor, _inputs|
        value = tensor.options[:container].read_value
        raise "variable #{tensor.options[:container].name} not initalized" if value.nil?

        value
      end

      register_op :stop_gradient, no_eval: true do |_context, _tensor, inputs|
        inputs[0]
      end

      register_op :assign, noop: true do |context, tensor, _inputs|
        assign = tensor.inputs[0] || tensor
        assign.container = global_eval(tensor, tensor.inputs[1], context)
        assign.container
      end

      register_op :assign_add, noop: true do |context, tensor, _inputs|
        assign = tensor.inputs[0] || tensor

        assign.container = process_vector_math_op(tensor, tensor.inputs[0], tensor.inputs[1], context) { |t, u| t + u }
        assign.container
      end

      register_op :assign_sub, noop: true do |context, tensor, _inputs|
        assign = tensor.inputs[0] || tensor

        assign.container = process_vector_math_op(tensor, tensor.inputs[0], tensor.inputs[1], context) { |t, u| t - u }
        assign.container
      end

      register_op :less do |context, tensor, inputs|
        a, b = inputs
        call_vector_op(tensor, :less, a, b, context) { |t, u| t < u }
      end

      register_op :greater do |context, tensor, inputs|
        a, b = inputs
        call_vector_op(tensor, :greater, a, b, context) { |t, u| t > u }
      end

      register_op :greater_equal do |context, tensor, inputs|
        a, b = inputs
        call_vector_op(tensor, :greater_equal, a, b, context) { |t, u| t >= u }
      end

      register_op :less_equal do |context, tensor, inputs|
        a, b = inputs
        call_vector_op(tensor, :greater_equal, a, b, context) { |t, u| t <= u }
      end

      register_op :broadcast_transform do |_context, _tensor, inputs|
        broadcast(inputs[0], inputs[1])
      end

      register_op :identity do |_context, _tensor, inputs|
        inputs[0]
      end

      register_op :print do |_context, tensor, inputs|
        puts "#{tensor.options.fetch(:message, "")} #{inputs[1]}"
        inputs[0]
      end

      register_op %i[div real_div], noop: true do |context, tensor, inputs|
        process_vector_math_op(tensor, inputs[0], inputs[1], context) { |t, u| t / u }
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

        inputs.shift
        variables = {}
        inputs.each do |savable|
          val = savable.container
          packed_data = Zlib::Deflate.deflate(TensorStream::Packer.pack(val, savable.data_type))
          variables[savable.name] = {
            "shape" => shape_eval(val),
            "data" => Base64.strict_encode64(packed_data),
          }
        end

        File.write(outputfile, {"variables" => variables}.to_yaml)
        nil
      end

      register_op :restore_ts do |_context, tensor, inputs|
        inputs = inputs.dup
        filename = inputs.shift
        tensor_names = inputs

        input_dump = YAML.safe_load(File.read(filename), [Symbol])
        vars = tensor.graph.get_collection(GraphKeys::GLOBAL_VARIABLES)

        vars.select! { |v| input_dump["variables"].key?(v.name) && tensor_names.include?(v.name) }
        vars.each do |variable|
          data = TensorStream::Packer.unpack(Zlib::Inflate.inflate(Base64.decode64(input_dump["variables"][variable.name]["data"])), variable.data_type)
          shape = input_dump["variables"][variable.name]["shape"]
          variable.buffer = nil
          variable.value = TensorShape.reshape(data, shape)
        end

        nil
      end

      register_op :check_numerics do |context, tensor, inputs|
        message = tensor.options[:message]
        call_op(inputs[0], context) do |t, _b|
          raise TensorStream::InvalidArgumentError, "#{message} Invalid argument" if t.nan? || t.infinite?

          t
        end
      end

      def eval_operation(tensor, child_context)
        return @context[tensor.name] if @context.key?(tensor.name)

        # puts "ruby eval #{object_id}: #{tensor.name}"
        invoke(tensor, child_context).tap do |result|
          # puts "result done ruby #{object_id}: #{tensor.name}"
          # assertions to make sure inferred shapes == actual evaluated shapes
          if tensor.shape.known? && (result.is_a?(Array) || result.is_a?(Float) || result.is_a?(Integer))
            raise "assert error #{tensor.name} #{shape_eval(result)} != #{tensor.shape.shape}" if shape_eval(result) != tensor.shape.shape
          end

          if tensor.breakpoint
            a = tensor.inputs[0] if tensor.inputs && tensor.inputs[0]
            b = tensor.inputs[1] if tensor.inputs && tensor.inputs[1]
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
              value: result,
            }
          end
          @context[tensor.name] = result
        end
      rescue EvaluatorExcecutionException => e
        raise e, "error #{e.message} while evaluating #{tensor.name}  defined at #{tensor.source}"
      rescue TensorStreamError => e
        raise e, "error #{e.message} while evaluating #{tensor.name}  defined at #{tensor.source}"
      rescue => e
        puts e.message
        puts e.backtrace.join("\n")
        raise EvaluatorExcecutionException.new(e, tensor), "error #{e.message} while evaluating #{tensor.name} : #{tensor.to_math(true, 1)} defined at #{tensor.source}"
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

      def call_op(a, child_context, &block)
        a = complete_eval(a, child_context)
        process_function_op(a, &block)
      end

      def call_vector_op(tensor, op, a, b, child_context, &block)
        process_vector_math_op(tensor, a, b, child_context, &block)
      rescue FullEvalNotPossible
        TensorStream.send(op.to_sym, a, b)
      end

      def process_vector_math_op(tensor, a, b, child_context, &block)
        eval_a = global_eval(tensor, a, child_context) unless a.nil?
        eval_b = global_eval(tensor, b, child_context) unless b.nil?

        raise FullEvalNotPossible.new, "full eval not possible for #{a.name}" if eval_a.is_a?(Tensor) || eval_b.is_a?(Tensor)

        # ruby scalar
        eval_a, eval_b = broadcast(eval_a, eval_b)
        vector_op(eval_a, eval_b, &block)
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
        if elem.is_a?(Array)
          elem.each_with_index.collect do |_item, index|
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

      # handle 3 tensor math operations
      def call_3way_vector_op(v_a, v_b, v_c, child_context, &block)
        return yield(v_a, v_b, v_c) unless v_a.is_a?(Array)

        v_a.each_with_index.collect do |v1, index|
          v2 = v_b[index]
          v3 = v_c.is_a?(Array) ? v_c[index] : v_c
          if v1.is_a?(Array)
            call_3way_vector_op(v1, v2, v3, child_context, &block)
          else
            yield(v1, v2, v3)
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
        File.write("/tmp/intermediates.txt", str)
      end
    end
  end
end

TensorStream::Evaluator.register_evaluator(TensorStream::Evaluator::RubyEvaluator, "ruby")
