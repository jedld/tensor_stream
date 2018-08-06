module TensorStream
  # Evaluator base module
  module Evaluator
    class OutputGroup
      attr_accessor :outputs
      def initialize(outputs = [])
        @outputs = outputs
      end
    end

    class UnsupportedOp < Exception
      def initialize(tensor)
        @tensor = tensor
      end

      def message
        "unsupported op #{@tensor.operation}"
      end
    end

    # Evaluator base class
    class BaseEvaluator
      def initialize(session, _device, thread_pool: nil, log_intermediates: false)
        @session = session
        @log_intermediates = log_intermediates
        @thread_pool = thread_pool || Concurrent::ImmediateExecutor.new
        @context[:compute_history] = [] if log_intermediates
      end

      ##
      # Query all supported devices
      def self.query_supported_devices
        [Device.new('cpu', :cpu, self)]
      end

      ##
      # Select the best device available in the system for this evaluator
      def self.default_device
        Device.new('cpu', :cpu, self)
      end

      ##
      # Selects the best device with the specified query, query can
      # be evaluator specific
      def self.fetch_device(_query = [])
        Device.new('cpu', :cpu, self)
      end

      ##
      # Select device using uri
      def self.query_device(query)
        return default_device if query.nil? || query == :default

        all_devices = query_supported_devices
        substrs = query.split('/')
        substrs.each do |q|
          components = q.split(':')
          next if components.size.zero?
          if components[0] == 'device' # use tensorflow convention
            device_type = components[1]
            select_index = components[2].to_i

            devices = all_devices.select { |d| d.type == device_type.downcase.to_sym }
            return nil if devices.empty?

            select_index = [devices.size - 1, select_index].min
            return devices[select_index]
          elsif components[0] == 'cpu'
            device_type = :cpu
            select_index = components[1].to_i

            devices = all_devices.select { |d| d.type == device_type.downcase.to_sym }
            return nil if devices.empty?

            select_index = [devices.size - 1, select_index].min
            return devices[select_index]
          elsif components[0] == 'ts' # tensorstream specific
            evaluator_class = TensorStream::Evaluator.evaluators[components[1]][:class]
            return nil unless self == evaluator_class
            return evaluator_class.fetch_device(components[2..components.size]) if evaluator_class.respond_to?(:fetch_device)
            return nil
          end
        end
      end

      ##
      # registers an op for the current evaluator class
      def self.register_op(opcode, options = {}, &block)
        @ops ||= {}
        if opcode.is_a?(Array)
          opcode.each do |op|
            @ops[op.to_sym] = { options: options, block: block }
          end
        else
          @ops[opcode.to_sym] = { options: options, block: block }
        end
      end

      ##
      # gets all supported ops for this Evaluator class
      def self.ops
        @ops ||= {}
      end

      def invoke(tensor, execution_context)
        return eval_tensor(tensor, execution_context) unless tensor.is_a?(Operation)
        raise UnsupportedOp.new(tensor), "op #{tensor.operation} is not yet supported" unless self.class.ops.key?(tensor.operation.to_sym)

        op = self.class.ops[tensor.operation.to_sym]
        op_options = op[:options]

        resolved_inputs = tensor.inputs.map do |i|
          next if i.nil?

          if i.is_a?(Array)
            next i.collect { |sub_item| sub_item.is_a?(Tensor) ? invoke(sub_item, execution_context) : sub_item }
          end

          if !op_options[:noop] && @context[:_cache][:placement][tensor.name] != @context[:_cache][:placement][i.name] # tensor is on another device or evaluator
            cache_key = "#{tensor.graph.object_id}_#{i.name}:#{object_id}"
            next @context[:_cache][cache_key] if @context[:_cache].key?(cache_key)

            result = @session.delegate_to_evaluator(i, @context, execution_context)
            convert_from_buffer(i, result).tap do |buffer|
              @context[:_cache][cache_key] = buffer if i.is_const
            end
          else
            prepare_input(i, execution_context, op_options)
          end
        end

        instance_exec(execution_context, tensor, resolved_inputs, &op[:block])
      end

      protected

      def get_broadcast_gradient_args(input_a, input_b)
        return [[], []] if input_a == input_b

        input_a_args = []
        input_b_args = []

        input_a = Array.new(input_b.size) { |i| i < input_a.size ? input_a[i] : nil }.reverse if input_a.size < input_b.size
        input_b = Array.new(input_a.size) { |i| i < input_b.size ? input_b[i] : nil }.reverse if input_a.size > input_b.size

        input_a.reverse.zip(input_b.reverse).each_with_index do |item, index|
          a, b = item

          if a.nil? || b && (a < b)
            input_a_args << input_b.size - index - 1
          elsif b.nil? || a && (a > b)
            input_b_args << input_a.size - index - 1
          end
        end

       [input_a_args.reverse, input_b_args.reverse]
      end

      ##
      # converts from a ruby Buffer object to the evaluator's native buffer format
      def convert_from_buffer(tensor, result)
        raise "need implementation"
      end

      def prepare_input(tensor, context, options = {})
        raise "need implementation"
      end
    end

    def self.evaluators
      @evaluators ||= {}
    end

    def self.register_evaluator(klass, name, index = 0)
      @evaluators ||= {}
      @evaluators[name] = { name: name, class: klass, index: index }
    end

    def self.default_evaluators
      evaluators.values.sort { |v| v[:index] }.reverse.map { |v| v[:class] }
    end
  end
end
