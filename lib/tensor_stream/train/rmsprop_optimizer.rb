module TensorStream
  module Train
    # High Level implementation of the RMSProp algorithm
    # This is a straight port from TensorFlows rmsprop.py
    class RMSPropOptimizer < Optimizer
      include TensorStream::OpHelper

      attr_accessor :learning_rate

      ##
      # Optimizer that implements the RMSProp algorithm.
      #
      # [paper](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
      def initialize(learning_rate, decay = 0.9, momentum = 0.0, epsilon = 1e-10, centered: false,
        use_locking: false, name: "RMSProp")
        @learning_rate = learning_rate
        @decay = decay
        @momentum = momentum
        @epsilon = epsilon
        @centered = centered

        # Tensor versions of the constructor arguments, created in _prepare().
        @learning_rate_tensor = nil
        @decay_tensor = nil
        @momentum_tensor = nil
        @epsilon_tensor = nil

        super(name: name, use_locking: use_locking)
      end

      protected

      def prepare
        lr = call_if_callable(@learning_rate)
        decay = call_if_callable(@decay)
        momentum = call_if_callable(@momentum)
        epsilon = call_if_callable(@epsilon)

        @learning_rate_tensor = TensorStream.convert_to_tensor(lr, name: "learning_rate")
        @decay_tensor = TensorStream.convert_to_tensor(decay, name: "decay")
        @momentum_tensor = TensorStream.convert_to_tensor(momentum, name: "momentum")
        @epsilon_tensor = TensorStream.convert_to_tensor(epsilon, name: "epsilon")
      end

      def create_slots(var_list)
        # Create slots for the first and second moments.
        var_list.each do |v|
          init_rms = if v.shape.known?
            TensorStream.ones_initializer(dtype: v.data_type)
          else
            TensorStream.ones_like(v)
          end

          get_or_make_slot_with_initializer(v, init_rms, v.shape, v.data_type, "rms", @name)

          zeros_slot(v, "mg", @name) if @centered
          zeros_slot(v, "momentum", @name)
        end
      end

      def apply_dense(grad, var)
        rms = get_slot(var, "rms")
        mom = get_slot(var, "momentum")

        if @centered
          mg = get_slot(var, "mg")
          _op(:apply_centered_rms_prop, var, mg, rms, mom,
            TensorStream.cast(@learning_rate_tensor, var.data_type),
            TensorStream.cast(@decay_tensor, var.data_type),
            TensorStream.cast(@momentum_tensor, var.data_type),
            TensorStream.cast(@epsilon_tensor, var.data_type),
            grad, use_locking: @use_locking)
        else
          _op(:apply_rms_prop, var, rms, mom,
            TensorStream.cast(@learning_rate_tensor, var.data_type),
            TensorStream.cast(@decay_tensor, var.data_type),
            TensorStream.cast(@momentum_tensor, var.data_type),
            TensorStream.cast(@epsilon_tensor, var.data_type),
            grad, use_locking: @use_locking)
        end
      end
    end
  end
end
