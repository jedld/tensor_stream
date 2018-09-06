module TensorStream
  class VariableScope
    attr_accessor :name, :reuse, :initializer
    attr_reader :used_names

    def initialize(name: nil, reuse: nil, initializer: nil)
      @name = name
      @reuse = reuse
      @initializer = initializer
      @used_names = []
    end

    def get_variable(name, dtype: nil, shape: nil, initializer: nil, trainable: true, collections: nil, validate_shape: false)
      raise TensorStream::ValueError, "validate_shape=true and initializer does not have a defined shape" if validate_shape && !shape.nil && initializer.is_a?(Tensor)
      TensorStream::Variable.new(dtype || :float32, nil, shape, self, collections: collections, name: name, initializer: initializer, trainable: trainable)
    end

    def register_name(name)
      @used_names << name unless @used_names.include?(name)
    end
  end
end