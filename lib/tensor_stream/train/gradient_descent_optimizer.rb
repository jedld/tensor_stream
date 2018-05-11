module TensorStream
  module Train
    # High Level implementation of the gradient descent algorithm
    class GradientDescentOptimizer
      attr_accessor :learning_rate

      def initialize(learning_rate, options = {})
        @learning_rate = learning_rate
      end

      def minimize(cost)
        trainable_vars = TensorStream::Graph.get_default_graph.
                                             get_collection(GraphKeys::GLOBAL_VARIABLES).
                                             select(&:trainable)

        derivatives = TensorStream.gradients(cost, trainable_vars)
        trainable_vars.each_with_index.collect do |var, index|
          var.assign_sub(derivatives[index] * @learning_rate)
        end
      end
    end
  end
end