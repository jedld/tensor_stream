module TensorStream
  module Train
    # High Level implementation of the gradient descent algorithm
    class GradientDescentOptimizer < Optimizer
      include TensorStream::OpHelper

      attr_accessor :learning_rate

      def initialize(learning_rate, use_locking: false, name: "GradientDescent")
        @learning_rate = learning_rate
        @learning_rate_tensor = nil
        super(name: name, use_locking: use_locking)
      end

      protected

      def prepare
        learning_rate = call_if_callable(@learning_rate)
        @learning_rate_tensor = convert_to_tensor(learning_rate, name: "learning_rate")
      end

      def apply_dense(grad, var)
        i_op(:apply_gradient_descent, var, TensorStream.cast(@learning_rate_tensor, grad.data_type), grad)
      end
    end
  end
end
