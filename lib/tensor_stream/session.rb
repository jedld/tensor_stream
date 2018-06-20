module TensorStream
  #  TensorStream class that defines a session
  class Session
    include StringHelper

    attr_reader :last_session_context, :closed, :target, :session_cache
    attr_accessor :randomizer

    def initialize(evaluator = :ruby_evaluator, thread_pool_class: Concurrent::ImmediateExecutor, evaluator_options: {})
      @evaluator_class = Object.const_get("TensorStream::Evaluator::#{camelize(evaluator.to_s)}")
      @thread_pool = thread_pool_class.new
      @closed = false
      @session_cache = {}
      @randomizer = {}
      @evaluator_options = evaluator_options
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
      evaluator = @evaluator_class.new(self, context.merge!(retain: options[:retain]), @evaluator_options)

      execution_context = {}
      @last_session_context = context
      result = args.collect { |e| evaluator.run(e, execution_context) }
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
      graph.nodes.select { |k, v| selector.call(k, v) }.collect do |k, node|
        next unless @last_session_context[node.name]
        "#{k} #{node.to_math(true, 1)} = #{@last_session_context[node.name]}"
      end.compact
    end

    def graph_ml(tensor, filename)
      TensorStream::Graphml.new(self).serialize(tensor, filename)
    end
  end
end
