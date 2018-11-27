module TensorStream
  # Class that defines a TensorStream variable
  class Variable < Tensor
    attr_accessor :trainable, :options, :buffer, :op
    def initialize(data_type, rank, shape, variable_scope, options = {})
      setup_initial_state(options)

      @options = {
      }
      @data_type = data_type
      @rank = rank
      @value = nil
      @is_const = false
      scope_name = variable_scope ? variable_scope.name : nil
      variable_scope_initializer = variable_scope ? variable_scope.initializer : nil
      @name = [scope_name, options[:name] || build_name].compact.reject(&:empty?).join('/')
      @initalizer_tensor = options[:initializer] || variable_scope_initializer || TensorStream.glorot_uniform_initializer
      shape = @initalizer_tensor.shape.shape if shape.nil? && @initalizer_tensor && @initalizer_tensor.shape

      @shape = TensorShape.new(shape, rank)
      @trainable = options.fetch(:trainable, true)
      @op = @graph.add_variable(self, options.merge(shape: @shape, data_type: @data_type))
    end

    def trainable?
      @trainable
    end

    def initializer
      init_op = @initalizer_tensor.op
      init_op.shape = @shape || init_op.shape
      init_op.data_type = @data_type || init_op.data_type
      assign(init_op)
    end

    def initialized_value
      init_op = @initalizer_tensor.op
      init_op.shape = @shape || init_op.shape
      init_op.data_type = @data_type || init_op.data_type
      init_op
    end

    def assign(value, name: nil, use_locking: false)
      _a, value = TensorStream.check_data_types(self, value)
      _op(:assign, self, value, name: name)
    end

    def read_value
      @value = buffer.to_ruby if buffer
      @value
    end

    def assign_add(value, name: nil)
      _a, value = TensorStream.check_data_types(self, value)
      _op(:assign_add, self, value, data_type: data_type, name: name)
    end

    def to_math(_tensor, _name_only = false, _max_depth = 99, _unused = 0)
      @name
    end

    def assign_sub(value)
      _a, value = TensorStream.check_data_types(self, value)
      _op(:assign_sub, self, value)
    end

    def self.variables_initializer(collection)
      TensorStream.group(TensorStream.get_default_graph.get_collection(collection).map(&:initializer))
    end

    def self.global_variables_initializer
      variables_initializer(TensorStream::GraphKeys::GLOBAL_VARIABLES)
    end
  end
end
