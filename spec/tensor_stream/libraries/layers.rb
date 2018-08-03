module TensorStream
  class Layers
    class Dense
      def initialize(inputs, units, activiation: nil, use_bias: true,
        kernel_initializer: nil,
        bias_initializer: ->() { TensorStream.zeros_initializer },
        trainable: true,
        name: nil)
      end
    end
  end
end