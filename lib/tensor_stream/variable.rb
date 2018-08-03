module TensorStream
  # Class that defines a TensorStream variable
  class Variable < Tensor
    attr_accessor :trainable, :options, :buffer
    def initialize(data_type, rank, shape, variable_scope, options = {})
      setup_initial_state(options)

      @options = {
      }
      @data_type = data_type
      @rank = rank
      @value = nil
      @is_const = false
      @name = [ variable_scope.name, options[:name] || build_name].compact.reject(&:empty?).join('/')
      @initalizer_tensor = options[:initializer] ? options[:initializer] : variable_scope.initializer || TensorStream.glorot_uniform_initializer
      if shape.nil? && @initalizer_tensor && @initalizer_tensor.shape
        shape = @initalizer_tensor.shape.shape
      end
      @shape = TensorShape.new(shape, rank)
      @trainable = options.fetch(:trainable, true)
      @graph.add_variable(self, options)
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

    def assign(value, name: nil)
      _a, value = TensorStream.check_data_types(self, value)
      Operation.new(:assign, self, value, name: name)
    end

    def read_value
      if buffer
        @value = buffer.to_ruby
      end

      @value
    end

    def assign_add(value)
      _a, value = TensorStream.check_data_types(self, value)
      Operation.new(:assign_add, self, value, data_type: data_type)
    end

    def to_math(_tensor, _name_only = false, _max_depth = 99, _unused = 0)
      @name
    end

    def assign_sub(value)
      _a, value = TensorStream.check_data_types(self, value)
      Operation.new(:assign_sub, self, value)
    end

    def self.variables_initializer(collection)
      TensorStream.group(TensorStream.get_default_graph.get_collection(collection).map(&:initializer))
    end

    def self.global_variables_initializer
      variables_initializer(TensorStream::GraphKeys::GLOBAL_VARIABLES)
    end
  end
end
