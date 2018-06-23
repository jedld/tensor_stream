module TensorStream
  #  TensorStream class that defines a session
  class Session
    include StringHelper

    attr_reader :last_session_context, :closed, :target, :session_cache
    attr_accessor :randomizer

    def initialize(evaluator = :ruby_evaluator, thread_pool_class: Concurrent::ImmediateExecutor, evaluator_options: {})
      @thread_pool = thread_pool_class.new
      @closed = false
      @session_cache = {}
      @randomizer = {}
      @evaluator_options = evaluator_options
      get_evaluator_classes(evaluator)
    end

    def get_evaluator_classes(evaluators)
      if evaluators.is_a?(Array)
        if evaluators.empty?
          @evaluator_classes = TensorStream::Evaluator::RubyEvaluator
        else
          @evaluator_classes = evaluators.collect { |name|  Object.const_get("TensorStream::Evaluator::#{camelize(name.to_s)}") }
        end
      else
        @evaluator_classes = [Object.const_get("TensorStream::Evaluator::#{camelize(evaluators.to_s)}")]
      end
    end

    def clear_session_cache
      @session_cache = {}
    end

    def self.default_session
      @session ||= Session.new
    end

    def run(*args)
      options = if args.last.is_a?(Hash)
                  args.pop
                else
                  {}
                end
      context = {
        _cache: @session_cache
      }

      # scan for placeholders and assign value
      if options[:feed_dict]
        options[:feed_dict].keys.each do |k|
          if k.is_a?(Placeholder)
            context[k.name.to_sym] = options[:feed_dict][k]
          end
        end
      end

      @evaluator_options[:thread_pool] = @thread_pool
      @evaluator_options[:log_intermediates] = options[:log_intermediates]

      args.each { |t| prepare_evaluators(t, context) }
      @last_session_context = context
      result = args.collect do |e|
        value = delegate_to_evaluator(e, context, {})
        value.respond_to?(:to_ruby) ? value.to_ruby : value
      end
      result.size == 1 ? result.first : result
    end

    def list_devices
      TensorStream::Evaluator.evaluators.collect do |k, v|
        v[:class].query_supported_devices.collect do |device|
          device
        end
      end.flatten
    end

    def close
      @closed = true
    end

    def closed?
      @closed
    end

    def dump_internal_ops(tensor)
      dump_ops(tensor, ->(_k, n) { n.is_a?(Tensor) && n.internal? })
    end

    def dump_user_ops(tensor)
      dump_ops(tensor, ->(_k, n) { n.is_a?(Tensor) && !n.internal? })
    end

    def dump_ops(tensor, selector)
      graph = tensor.graph
      graph.nodes.select { |k, v| selector.call(k, v) }.collect { |k, node|
        next unless @last_session_context[node.name]
        "#{k} #{node.to_math(true, 1)} = #{@last_session_context[node.name]}"
      }.compact
    end

    def graph_ml(tensor, filename)
      TensorStream::Graphml.new(self).serialize(tensor, filename)
    end

    def delegate_to_evaluator(tensor_arr, session_context, context)
      arr = tensor_arr.is_a?(Array) ? tensor_arr : [tensor_arr]
      result = arr.collect do |tensor|
        session_context[:placement][tensor.name].run_with_buffer(tensor, session_context, context)
      end
      result.size == 1 ? result.first : result
    end

    protected

    def assign_evaluator(tensor, evaluators)
      evaluators.each do |evaluator|
        next if tensor.is_a?(Operation) && !evaluator.class.ops.include?(tensor.operation.to_sym)
        return evaluator
      end
      raise "no evaluator available to execute #{tensor.operation}"
    end

    def prepare_evaluators(tensor_arr, context)
      context[:placement] = {}
      evaluators = @evaluator_classes.map { |klass| klass.new(self, @evaluator_options) }
      tensor_arr = tensor_arr.is_a?(Array) ? tensor_arr : [tensor_arr]
      tensor_arr.each do |tensor|
        graph = tensor.graph
        graph.nodes.values.each do |node|
          context[:placement][node.name] = assign_evaluator(node, evaluators)
        end
      end
    end
  end
end
