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

    ##
    # List available evaluators + devices in the current local environment
    # Returns:
    # - An array containing the names of those devices
    def list_local_devices
      local_name = 'job:localhost'
      TensorStream::Evaluator.evaluators.collect do |k, v|
        v[:class].query_supported_devices.collect do |device_str|
          [local_name, "ts:#{k}:#{device_str.name}"].join('/')
        end
      end.flatten
    end

    ##
    # Creates a variable
    # A variable maintains state across sessions
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
                 TensorStream::Variable.new(dtype || :string, 0, [], get_variable_scope, common_options)
               elsif value.is_a?(Integer)
                 TensorStream::Variable.new(dtype || :int32, 0, [], get_variable_scope, common_options)
               elsif value.is_a?(Float)
                 TensorStream::Variable.new(dtype || :float32, 0, [], get_variable_scope, common_options)
               else
                 TensorStream::Variable.new(dtype || :float32, 0, nil, get_variable_scope, common_options)
               end
      op.inputs[0] = tensor
      tensor
    end

    ##
    # Defines a variable context manager
    def variable_scope(scope = nil, default_name = nil, reuse: nil, initializer: nil)
      Thread.current[:tensor_stream_variable_scope] ||= [ VariableScope.new ]

      # uniquenifier
      if scope.nil? && default_name
        same_names = get_variable_scope.used_names.select { |s| s.start_with?(default_name) }
        new_name = default_name
        index = 1
        while same_names.include?(new_name)
          new_name = "#{default_name}_#{index}"
          index += 1
        end
        scope = new_name
      end

      variable_scope = VariableScope.new(name: scope, reuse: reuse, initializer: initializer)
      get_variable_scope.register_name(scope || "")
      Thread.current[:tensor_stream_variable_scope] << variable_scope
      scope_name = __v_scope_name
      if block_given?
        begin
          TensorStream.get_default_graph.name_scope(scope) do
            yield(scope_name)
          end
        ensure
          Thread.current[:tensor_stream_variable_scope].pop
        end
      else
        variable_scope
      end
    end

    def device(device_uri, &block)
      get_default_graph.device(device_uri, &block)
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
      if !Thread.current[:tensor_stream_variable_scope]
        variable_scope = VariableScope.new
        Thread.current[:tensor_stream_variable_scope] = [variable_scope]
        return variable_scope
      end

      Thread.current[:tensor_stream_variable_scope].last
    end

    def __v_scope_name
      Thread.current[:tensor_stream_variable_scope].map(&:name).compact.reject(&:empty?).join('/')
    end

    def session(evaluator = nil, thread_pool_class: Concurrent::ImmediateExecutor, log_device_placement: false)
      session = TensorStream::Session.new(evaluator, thread_pool_class: thread_pool_class, log_device_placement: log_device_placement)
      yield session if block_given?

      session
    end

    def colocate_with(op, ignore_existing: false)
      # noop for now
      yield
    end

    def program
      yield self
    end

    def layers
      TensorStream::Layers
    end

    def constant(value, dtype: nil, shape: nil, internal: false, name: 'Const')
      shared_options = { const: true, value: value, name: name, internal: internal }
      if value.is_a?(Float)
        TensorStream::Tensor.new(dtype || :float32, 0, shape || [], shared_options)
      elsif value.is_a?(Integer)
        TensorStream::Tensor.new(dtype || :int32, 0, shape || [], shared_options)
      elsif value.is_a?(String)
        TensorStream::Tensor.new(dtype || :string, 0, shape || [], shared_options)
      elsif !!value == value
        TensorStream::Tensor.new(dtype || :boolean, 0, shape || [], shared_options)
      elsif value.is_a?(Array)
        dimension = shape || shape_eval(value)
        rank = dimension.size

        cur_dtype = dtype || Tensor.detect_type(value.flatten.last)
        value = Tensor.cast_dtype(value, cur_dtype) unless dtype.nil?

        shared_options[:value] = value
        TensorStream::Tensor.new(cur_dtype, rank, dimension, shared_options)
      end
    end

    def group(inputs, name: nil)
      TensorStream::ControlFlow.new(:group, inputs, nil, name: name)
    end

    def dynamic_stitch(indices, data, name: nil)
      TensorStream::DynamicStitch.new(:dynamic_stitch, [indices, data], name: name)
    end

    def get_variable(name, dtype: nil, shape: nil, initializer: nil, trainable: true, collections: nil, validate_shape: false)
      get_variable_scope.get_variable(name, dtype: dtype, shape: shape, initializer: initializer, trainable: trainable, collections: collections)
    end

    def get_collection(name, options = {})
      Graph.get_default_graph.get_collection(name, options)
    end

    def assign(ref, value, name: nil)
      raise "#{ref.name} not a variable" unless ref.is_a?(Variable)
      ref.assign(value, name: name)
    end

    ##
    # Inserts a placeholder for a tensor that will be always fed.
    def placeholder(dtype, shape: nil, name: nil)
      TensorStream::Placeholder.new(dtype, nil, shape, name: name)
    end

    def global_variables_initializer
      TensorStream::Variable.global_variables_initializer
    end

    def train
      TensorStream::Trainer
    end

    def image
      TensorStream::Images
    end

    def trainable_variables
      TensorStream.get_default_graph.get_collection(TensorStream::GraphKeys::TRAINABLE_VARIABLES)
    end

    ##
    # Sets random seed to use for the default graph
    def set_random_seed(seed)
      TensorStream.get_default_graph.random_seed = seed
    end

    def control_dependencies(control_inputs, &block)
      TensorStream.get_default_graph.control_dependencies(control_inputs, &block)
    end

    def convert_to_tensor(value, dtype: nil, name: nil, preferred_dtype: nil)
      return value if value.is_a?(Tensor)
      return convert_to_tensor(value.call) if value.is_a?(Proc)
      if value.is_a?(Array) && value[0].is_a?(Tensor)
        if value.size > 1
          return TensorStream.concat(value.map { |v| TensorStream.expand_dims(v, 0) }, 0)
        else
          return TensorStream.expand_dims(value[0], 0)
        end
      end

      i_cons(value, dtype: dtype || Tensor.detect_type(value), name: name)
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
        raise TensorStream::ValueError, "Value Error: Tensor conversion requested dtype #{input_a.data_type} for tensor type #{input_b.data_type}"
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