module TensorStream
  #  TensorStream class that defines a session
  class Session
    include StringHelper

    attr_reader :last_session_context, :closed, :target, :session_cache
    attr_accessor :randomizer

    def initialize(evaluator = nil, thread_pool_class: Concurrent::ImmediateExecutor, log_device_placement: false, evaluator_options: {})
      @thread_pool = thread_pool_class.new
      @closed = false
      @session_cache = {}
      @randomizer = {}
      @log_device_placement = log_device_placement
      @evaluator_options = evaluator_options
      get_evaluator_classes(evaluator)
      @evaluators = {}
    end

    def get_evaluator_classes(evaluators)
      @evaluator_classes = if evaluators.is_a?(Array)
                             if evaluators.empty?
                               TensorStream::Evaluator.default_evaluators
                             else
                               evaluators.collect { |name|  Object.const_get("TensorStream::Evaluator::#{camelize(name.to_s)}") }
                             end
                           elsif evaluators.nil?
                             TensorStream::Evaluator.default_evaluators
                           else
                             [Object.const_get("TensorStream::Evaluator::#{camelize(evaluators.to_s)}")]
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

      if @log_device_placement
        context[:_cache][:placement].each do |k, v|
          puts "#{k} : #{v[0].name}"
        end
      end
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
        session_context[:_cache][:placement][tensor.name][1].run_with_buffer(tensor, session_context, context)
      end
      result.size == 1 ? result.first : result
    end

    protected

    def assign_evaluator(tensor)
      device = @evaluator_classes.map do |klass|
        next nil if tensor.is_a?(Operation) && !klass.ops.include?(tensor.operation.to_sym)
        next klass.default_device if tensor.device.nil?

        klass.query_device(tensor.device)
      end.compact.first

      raise "no evaluator available to execute #{tensor.operation}" if device.nil?

      key = "#{device.evaluator.to_s}/#{device.name}"
      if @evaluators.key?(key)
        @evaluators[key]
      else
        @evaluators[key] = [device, device.evaluator.new(self, device)]
      end
    end

    def prepare_evaluators(tensor_arr, context)
      context[:_cache][:placement] ||= {}

      tensor_arr = tensor_arr.is_a?(Array) ? tensor_arr : [tensor_arr]
      tensor_arr.each do |tensor|
        next if context[:_cache][:placement][tensor.name]

        graph = tensor.graph
        graph.nodes.values.each do |node|
          context[:_cache][:placement][node.name] = assign_evaluator(node)
        end
      end
    end
  end
end
