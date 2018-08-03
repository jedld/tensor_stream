module TensorStream
  class VariableScope
    attr_accessor :name, :reuse, :initializer

    def initialize(name: '', reuse: nil, initializer: nil)
      @name = name
      @reuse = reuse
      @initializer = initializer
    end

    def get_variable(name, dtype: nil, shape: nil, initializer: nil, trainable: true, collections: nil)
      TensorStream::Variable.new(dtype || :float32, nil, shape, self, collections: collections, name: name, initializer: initializer, trainable: trainable)
    end
  end
end