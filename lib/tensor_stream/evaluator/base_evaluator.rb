module TensorStream
  module Evaluator
    class UnsupportedOp < Exception
      def initialize(tensor)
        @tensor = tensor
      end

      def message
        "unsupported op #{tensor.operation}"
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

      def self.register_op(opcode, &block)
        @ops ||= {}
        @ops[opcode.to_sym] = block
      end

      def invoke(tensor, context)
        if @ops.key?(tensor.operation.to_sym)
          resolved_inputs = tensor.inputs.map { |i| eval(resolve_placeholder(i)) if i }
          @ops[tensor.operation.to_sym].call(context, tensor, resolved_inputs)
        else
          raise UnsupportedOp.new(tensor)
        end
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
