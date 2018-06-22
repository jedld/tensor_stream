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
      def initialize(session, context, thread_pool: nil, log_intermediates: false)
        @session = session
        @context = context
        @log_intermediates = log_intermediates
        @thread_pool = thread_pool || Concurrent::ImmediateExecutor.new
        @context[:compute_history] = [] if log_intermediates
      end

      def self.query_supported_devices
        [Device.new("cpu", :cpu, "ruby")]
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

      def invoke(tensor, context)
        if self.class.ops.key?(tensor.operation.to_sym)
          op = self.class.ops[tensor.operation.to_sym]
          op_options = op[:options]
          resolved_inputs = tensor.inputs.map { |i| prepare_input(i, context, op_options)}
          instance_exec(context, tensor, resolved_inputs, &op[:block])
        else
          raise UnsupportedOp.new(tensor)
        end
      end

      protected

      def prepare_input(tensor, context, options = {})
        raise "need implementation"
      end
    end

    def self.evaluators
      @evaluators ||= {}
    end

    def self.register_evaluator(klass, name)
      @evaluators ||= {}
      @evaluators[name] = { name: name, class: klass }
    end
  end
end
