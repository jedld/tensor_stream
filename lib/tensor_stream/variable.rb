module TensorStream
  # Class that defines a TensorStream variable
  class Variable < Tensor
    attr_accessor :trainable
    def initialize(data_type, rank, shape, options = {})
      @graph = options[:graph] || TensorStream.get_default_graph

      @data_type = data_type
      @rank = rank
      @shape = TensorShape.new(shape, rank)
      @value = nil
      @source = format_source(caller_locations)
      @name = options[:name] || build_name
      @initalizer_tensor = options[:initializer] if options[:initializer]
      @trainable = options.fetch(:trainable, true)
      @graph.add_variable(self, options)
    end

    def trainable?
      @trainable
    end

    def initializer
      @initalizer_tensor.shape = @shape
      assign(@initalizer_tensor)
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

    def to_math(_tensor, _name_only = false, _max_depth = 99)
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
  end
end
