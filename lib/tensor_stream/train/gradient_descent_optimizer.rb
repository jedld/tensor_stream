module TensorStream
  module Train
    # High Level implementation of the gradient descent algorithm
    class GradientDescentOptimizer
      attr_accessor :learning_rate

      def initialize(learning_rate, _options = {})
        @learning_rate = learning_rate
      end

      def minimize(cost)
        trainable_vars = TensorStream.trainable_variables
        derivatives = TensorStream.gradients(cost, trainable_vars)
        trainable_vars.each_with_index.collect do |var, index|
          var.assign_sub(derivatives[index] * @learning_rate)
        end
      end
    end
  end
end
