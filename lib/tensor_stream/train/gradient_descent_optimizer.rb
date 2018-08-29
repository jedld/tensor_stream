module TensorStream
  module Train
    # High Level implementation of the gradient descent algorithm
    class GradientDescentOptimizer < Optimizer
      include TensorStream::OpHelper

      attr_accessor :learning_rate

      def initialize(learning_rate, use_locking: false, name: "GradientDescent")
        @learning_rate = learning_rate
        super(name: name, use_locking: use_locking)
      end

      protected

      def apply_dense(grad, var)
        i_op(:apply_gradient_descent, var, TensorStream.cast(@learning_rate, grad.data_type), grad)
      end
    end
  end
end
