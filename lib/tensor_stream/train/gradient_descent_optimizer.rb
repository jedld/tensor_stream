module TensorStream
  module Train
    # High Level implementation of the gradient descent algorithm
    class GradientDescentOptimizer < Optimizer
      include TensorStream::OpHelper

      attr_accessor :learning_rate

      def initialize(learning_rate, _options = {})
        @learning_rate = learning_rate
      end

      protected

      def apply_dense(grad, var)
        i_op(:apply_gradient_descent, var, TensorStream.cast(@learning_rate, grad.data_type), grad)
      end
    end
  end
end
