module TensorStream
  module Utils
    def float32
      Types.float32
    end

    def graph
      TensorStream::Graph.new
    end

    def get_default_graph
      TensorStream::Graph.get_default_graph
    end

    def reset_default_graph
      TensorStream::Graph.get_default_graph.reset
    end

    def enable_eager_execution
      TensorStream::Graph.get_default_graph.enable_eager_execution
    end

    def disable_eager_execution
      TensorStream::Graph.get_default_graph.disable_eager_execution
    end

    def executing_eagerly?
      TensorStream::Graph.get_default_graph.executing_eagerly?
    end

    def variable(value, name: nil, initializer: nil, graph: nil, dtype: nil, trainable: true)
      op = Operation.new(:assign, nil, value)
      common_options = {
        initializer: initializer || op,
        name: name,
        graph: graph,
        dtype: dtype,
        trainable: trainable
      }
      tensor = if value.is_a?(String)
        TensorStream::Variable.new(dtype || :string, 0, [], common_options)
      elsif value.is_a?(Integer)
        TensorStream::Variable.new(dtype || :int32, 0, [], common_options)
      elsif value.is_a?(Float)
        TensorStream::Variable.new(dtype || :float32, 0, [], common_options)
      else
        TensorStream::Variable.new(dtype || :float32, 0, nil, common_options)
      end
      op.items[0] = tensor
      tensor
    end

    def variable_scope(scope = nil, reuse: nil, initializer: nil)
      Thread.current[:tensor_stream_variable_scope] ||= []
      Thread.current[:tensor_stream_variable_scope] << OpenStruct.new(name: scope, reuse: reuse, initializer: initializer)
      scope_name = __v_scope_name
      begin
        if block_given?
          TensorStream.get_default_graph.name_scope(scope) do
            yield(scope_name)
          end
        end
      ensure
        Thread.current[:tensor_stream_variable_scope].pop
      end
    end

    def name_scope(name, default: nil, values: nil)
      if values
        graph_count = values.select { |v| v.is_a?(Tensor) }.map(&:graph).map(&:object_id).uniq.size
        raise "values are not on the same graph" if graph_count > 1
      end

      get_default_graph.name_scope(name || default) do |scope|
        yield scope if block_given?
      end
    end

    def get_variable_scope
      return nil unless Thread.current[:tensor_stream_variable_scope]
      __v_scope_name
    end

    def __v_scope_name
      Thread.current[:tensor_stream_variable_scope].map(&:name).compact.reject(&:empty?).join('/')
    end

    def session(evaluator = :ruby_evaluator, thread_pool_class: Concurrent::ImmediateExecutor)
      session = TensorStream::Session.new(evaluator, thread_pool_class: thread_pool_class)
      yield session if block_given?

      session
    end

    def program(&block)
      block.call(self)
    end

    def layers
      TensorStream::Layers
    end

    def constant(value, options = {})
      shared_options = { const: true, value: value, name: options[:name] }
      if value.is_a?(Float)
        TensorStream::Tensor.new(options[:dtype] || :float32, 0, options[:shape] || [], shared_options)
      elsif value.is_a?(Integer)
        TensorStream::Tensor.new(options[:dtype] || :int32, 0, options[:shape] || [], shared_options)
      elsif value.is_a?(String)
        TensorStream::Tensor.new(options[:dtype] || :string, 0, options[:shape] || [], shared_options)
      elsif value.is_a?(Array)
        dtype = nil
        rank = 1
        dimensions = []
        value_ptr = value

        Kernel.loop do
          dtype, rank, value_ptr, d = dtype_eval(rank, value_ptr)
          dimensions << d
          break if dtype != :array
        end

        TensorStream::Tensor.new(dtype, rank, options[:shape] || dimensions, shared_options)
      end
    end

    def group(inputs, name: nil)
      TensorStream::ControlFlow.new(:group, inputs, nil, name: name)
    end

    def get_variable(name, dtype: nil, shape: nil, initializer: nil, trainable: true, collections: nil)
      TensorStream::Variable.new(dtype || :float32, nil, shape, collections: collections, name: name, initializer: initializer, trainable: trainable)
    end

    def get_collection(name, options = {})
      Graph.get_default_graph.get_collection(name, options)
    end

    def placeholder(dtype, shape: nil, name: nil)
      TensorStream::Placeholder.new(dtype, nil, shape, name: name)
    end

    def global_variables_initializer
      TensorStream::Variable.global_variables_initializer
    end

    def train
      TensorStream::Trainer
    end

    def trainable_variables
      TensorStream.get_default_graph.get_collection(TensorStream::GraphKeys::TRAINABLE_VARIABLES)
    end

    def set_random_seed(seed)
      TensorStream.get_default_graph.random_seed = seed
    end

    def convert_to_tensor(value, dtype: nil, name: nil, preferred_dtype: nil)
      return convert_to_tensor(value.call) if value.is_a?(Proc)

      if !value.is_a?(Tensor)
        i_cons(value, dtype: dtype || Tensor.detect_type(value), name: name)
      else
        value
      end
    end

    def check_allowed_types(input, types)
      return input unless input.is_a?(Tensor)
      return input if input.data_type.nil?

      raise "#{input.source}: Parameter data type #{input.data_type} passed not in #{types.join(',')}" unless types.include?(input.data_type.to_sym)
    end

    def check_data_types(input_a, input_b)
      if !input_a.is_a?(Tensor) && input_b.is_a?(Tensor)
        input_a = convert_to_tensor(input_a, dtype: input_b.data_type)
      elsif !input_b.is_a?(Tensor) && input_a.is_a?(Tensor)
        input_b = convert_to_tensor(input_b, dtype: input_a.data_type)
      else
        input_a = convert_to_tensor(input_a)
        input_b = convert_to_tensor(input_b)
      end

      if norm_dtype(input_a.data_type) != norm_dtype(input_b.data_type)
        raise "Value Error: Tensor conversion requested dtype #{input_a.data_type} for tensor type #{input_b.data_type}" 
      end

      [input_a, input_b]
    end

    def norm_dtype(dtype)
      dtype = dtype.to_sym
      case dtype
      when :int
        :int32
      when :float
        :float32
      else
        dtype
      end
    end
  end
end