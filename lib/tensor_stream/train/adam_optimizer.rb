module TensorStream
  module Train
    # High Level implementation of the ADAM algorithm
    class AdamOptimizer < Optimizer
      include TensorStream::OpHelper

      attr_accessor :learning_rate

      ##
      # Construct a new Adam optimizer.
      #
      # Args:
      # learning_rate: A Tensor or a floating point value.  The learning rate.
      # beta1: A float value or a constant float tensor.
      #   The exponential decay rate for the 1st moment estimates.
      # beta2: A float value or a constant float tensor.
      #   The exponential decay rate for the 2nd moment estimates.
      # epsilon: A small constant for numerical stability. This epsilon is
      #   "epsilon hat" in the Kingma and Ba paper (in the formula just before
      #   Section 2.1), not the epsilon in Algorithm 1 of the paper.
      # use_locking: If True use locks for update operations.
      # name: Optional name for the operations created when applying gradients.
      #   Defaults to "Adam".
      def initialize(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
                     use_locking: false, name: "Adam")
        @learning_rate = learning_rate
        @beta1 = beta1
        @beta2 = beta2
        @epsilon = epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        @lr_t = nil
        @beta1_t = nil
        @beta2_t = nil
        @epsilon_t = nil

        # Created in SparseApply if needed.
        @updated_lr = nil
        super(name: name, use_locking: use_locking)
      end

      protected

      def get_beta_accumulators
        graph = TensorStream.get_default_graph
        [get_non_slot_variable("beta1_power", graph: graph),
         get_non_slot_variable("beta2_power", graph: graph)]
      end

      def prepare
        lr = call_if_callable(@learning_rate)
        beta1 = call_if_callable(@beta1)
        beta2 = call_if_callable(@beta2)
        epsilon = call_if_callable(@epsilon)

        @lr_t = TensorStream.convert_to_tensor(lr, name: "learning_rate")
        @beta1_t = TensorStream.convert_to_tensor(beta1, name: "beta1")
        @beta2_t = TensorStream.convert_to_tensor(beta2, name: "beta2")
        @epsilon_t = TensorStream.convert_to_tensor(epsilon, name: "epsilon")
      end

      def create_slots(var_list)
        first_var = var_list.min_by(&:name)
        create_non_slot_variable(@beta1, "beta1_power", first_var)
        create_non_slot_variable(@beta2, "beta2_power", first_var)

        # Create slots for the first and second moments.
        var_list.each do |v|
          zeros_slot(v, "m", @name)
          zeros_slot(v, "v", @name)
        end
      end

      def apply_dense(grad, var)
        m = get_slot(var, "m")
        v = get_slot(var, "v")
        beta1_power, beta2_power = get_beta_accumulators
        _op(:apply_adam,
            var, m, v,
            TensorStream.cast(beta1_power, var.data_type),
            TensorStream.cast(beta2_power, var.data_type),
            TensorStream.cast(@lr_t, var.data_type),
            TensorStream.cast(@beta1_t, var.data_type),
            TensorStream.cast(@beta2_t, var.data_type),
            TensorStream.cast(@epsilon_t, var.data_type),
            grad, use_locking: @use_locking)
      end

      def finish(update_ops, name_scope)
        TensorStream.control_dependencies(update_ops) do
          beta1_power, beta2_power = get_beta_accumulators
          update_beta1 = nil, update_beta2 = nil
          TensorStream.colocate_with(beta1_power) do
            update_beta1 = beta1_power.assign(beta1_power * @beta1_t, use_locking: @use_locking)
            update_beta2 = beta2_power.assign(beta2_power * @beta2_t, use_locking: @use_locking)
          end
          TensorStream.group(update_ops + [update_beta1, update_beta2], name: name_scope)
        end
      end
    end
  end
end