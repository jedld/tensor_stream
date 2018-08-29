module TensorStream
  module Train
    # Optimizer that implements the Momentum algorithm. loosely based on the tensorflow implementation.
    class MomentumOptimizer < Optimizer
      include OpHelper

      ##
      # Construct a new Momentum optimizer.
      #
      # Args:
      #   learning_rate: A Tensor or a floating point value that indicates the learning rate
      #   momentum: A Tensor or a floating point value for the momentum
      #   name: Optional name prefix
      #   use_nesterov: boolean - Flag that indicates if nesterov momentum is to be used. http://jmlr.org/proceedings/papers/v28/sutskever13.pdf
      #   use_locking: boolean - filler argument for compatibility, not used at the moment
      def initialize(learning_rate, momentum, name: 'momentum', use_nesterov: false, use_locking: false)
        @learning_rate = learning_rate
        @momentum = momentum
        @use_nesterov = use_nesterov
        super(name: name, use_locking: use_locking)
      end

      protected

      def prepare
        @learning_rate_tensor = TensorStream.convert_to_tensor(@learning_rate, name: "learning_rate")
        @momentum_tensor = TensorStream.convert_to_tensor(@momentum, name: "momentum")
      end

      def create_slots(var_list)
        var_list.each do |v|
          zeros_slot(v, "momentum", @name)
        end
      end

      def apply_dense(grad, var)
        mom = get_slot(var, "momentum")

        _op(:apply_momentum, var, mom,
            TensorStream.cast(@learning_rate_tensor, var.data_type),
            grad,
            TensorStream.cast(@momentum_tensor, var.data_type),
            use_locking: @use_locking,
            use_nesterov: @use_nesterov)
      end
    end
  end
end