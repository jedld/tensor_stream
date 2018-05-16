module TensorStream
  #  TensorStream class that defines a session
  class Session
    attr_reader :last_session_context, :closed, :target
    attr_accessor :randomizer
  
    def initialize(evaluator = :ruby_evaluator, thread_pool_class: Concurrent::ImmediateExecutor)
      @evaluator_class = Object.const_get("TensorStream::Evaluator::#{camelize(evaluator.to_s)}")
      @thread_pool = thread_pool_class.new
      @closed = false
      @randomizer = {}
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
      context = {}

      # scan for placeholders and assign value
      if options[:feed_dict]
        options[:feed_dict].keys.each do |k|
          context[k.name.to_sym] = options[:feed_dict][k] if k.is_a?(Placeholder)
        end
      end

      evaluator = @evaluator_class.new(self, context.merge!(retain: options[:retain]), thread_pool: @thread_pool)

      execution_context = {}
      result = args.collect { |e| evaluator.run(e, execution_context) }
      @last_session_context = context
      result.size == 1 ? result.first : result
    end

    def list_devices
      [Device.new("cpu")]
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

    private

    def camelize(string, uppercase_first_letter = true)
      string = if uppercase_first_letter
                 string.sub(/^[a-z\d]*/) { $&.capitalize }
               else
                 string.sub(/^(?:(?=\b|[A-Z_])|\w)/) { $&.downcase }
               end
      string.gsub(/(?:_|(\/))([a-z\d]*)/) { "#{$1}#{$2.capitalize}" }.gsub('/', '::')
    end
  end
end
