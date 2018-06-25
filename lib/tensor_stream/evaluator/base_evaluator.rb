module TensorStream
  module Evaluator
    class UnsupportedOp < Exception
      def initialize(tensor)
        @tensor = tensor
      end

      def message
        "unsupported op #{@tensor.operation}"
      end
    end

    class BaseEvaluator
      def initialize(session, device, thread_pool: nil, log_intermediates: false)
        @session = session
        @log_intermediates = log_intermediates
        @thread_pool = thread_pool || Concurrent::ImmediateExecutor.new
        @context[:compute_history] = [] if log_intermediates
      end

      def self.query_supported_devices
        [Device.new("cpu", :cpu, self)]
      end

      ##
      # Select the best device available in the system for this evaluator
      def self.default_device
        Device.new("cpu", :cpu, self)
      end

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
          end
        end
      end

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

      def self.ops
        @ops ||={}
      end

      def invoke(tensor, execution_context)
        if self.class.ops.key?(tensor.operation.to_sym)
          op = self.class.ops[tensor.operation.to_sym]
          op_options = op[:options]
          resolved_inputs = tensor.inputs.map do |i|
            next if i.nil?
            if @context[:_cache][:placement][tensor.name] != @context[:_cache][:placement][i.name] # tensor is on another device or evaluator
              result = @session.delegate_to_evaluator(i, @context, execution_context)
              convert_from_buffer(i, result)
            else
              prepare_input(i, execution_context, op_options)
            end
          end
          instance_exec(execution_context, tensor, resolved_inputs, &op[:block])
        else
          raise UnsupportedOp.new(tensor)
        end
      end

      protected

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
