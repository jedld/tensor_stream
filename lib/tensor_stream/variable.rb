module TensorStream
  # Class that defines a TensorStream variable
  class Variable < Tensor
    attr_accessor :trainable, :options
    def initialize(data_type, rank, shape, options = {})
      @graph = options[:graph] || TensorStream.get_default_graph
      @options = {
        
      }
      @data_type = data_type
      @rank = rank
      @value = nil
      @source = format_source(caller_locations)
      @name = [TensorStream.get_variable_scope, options[:name] || build_name].compact.reject(&:empty?).join('/')
      @initalizer_tensor = options[:initializer] ? options[:initializer] : _variable_scope.initializer || TensorStream.glorot_uniform_initializer
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

    def assign(value)
      Operation.new(:assign, self, value)
    end

    def read_value
      @value
    end

    def assign_add(value)
      Operation.new(:assign_add, self, value)
    end

    def to_math(_tensor, _name_only = false, _max_depth = 99, _unused = 0)
      @name
    end

    def assign_sub(value)
      Operation.new(:assign_sub, self, value)
    end

    def self.variables_initializer(collection)
      TensorStream.group(TensorStream.get_default_graph.get_collection(collection).map(&:initializer))
    end

    def self.global_variables_initializer
      variables_initializer(TensorStream::GraphKeys::GLOBAL_VARIABLES)
    end

    private

    def _variable_scope
      return OpenStruct.new(name: '', reuse: false, initializer: nil) if Thread.current[:tensor_stream_variable_scope].nil? || Thread.current[:tensor_stream_variable_scope].empty?
      scope = Thread.current[:tensor_stream_variable_scope].last
      scope
    end
  end
end
