module TensorStream
  module Train
    # High Level implementation of the Adadelta algorithm
    class AdadeltaOptimizer < Optimizer
      include TensorStream::OpHelper

      attr_accessor :learning_rate

      def initialize(learning_rate = 0.001, rho = 0.95, epsilon = 1e-8,
        use_locking: false, name: "Adadelta")
        @learning_rate = learning_rate
        @rho = rho
        @epsilon = epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        @learning_rate_tensor = nil
        @rho_t = nil
        @epsilon_t = nil
        super(name: name, use_locking: use_locking)
      end

      protected

      def create_slots(var_list)
        var_list.each do |v|
          zeros_slot(v, "accum", @name)
          zeros_slot(v, "accum_update", @name)
        end
      end

      def prepare
        @learning_rate_tensor = convert_to_tensor(@learning_rate, name: "lr")
        @rho_t = convert_to_tensor(@rho, name: "rho")
        @epsilon_t = convert_to_tensor(@epsilon, name: "epsilon")
      end

      def apply_dense(grad, var)
        accum = get_slot(var, "accum")
        accum_update = get_slot(var, "accum_update")
        _op(:apply_adadelta,
            var,
            accum,
            accum_update,
            TensorStream.cast(@learning_rate_tensor, var.data_type),
            TensorStream.cast(@rho_t, var.data_type),
            TensorStream.cast(@epsilon_t, var.data_type),
            grad,
            use_locking: @use_locking)
      end
    end
  end
end
