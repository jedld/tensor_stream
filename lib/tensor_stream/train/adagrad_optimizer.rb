module TensorStream
  module Train
    # High Level implementation of the Adagrad algorithm
    class AdagradOptimizer < Optimizer
      include TensorStream::OpHelper

      attr_accessor :learning_rate

      def initialize(learning_rate, initial_accumulator_value = 0.1,
        use_locking: false, name: "Adagrad")
        @learning_rate = learning_rate
        @initial_accumulator_value = initial_accumulator_value
        @learning_rate_tensor = nil
        super(name: name, use_locking: use_locking)
      end

      protected

      def create_slots(var_list)
        var_list.each do |v|
          dtype = v.data_type
          init = nil
          if v.shape.known?
            init = TensorStream.constant_initializer(@initial_accumulator_value, dtype: dtype)
          else
            init_constant = TensorStream.fill(TensorStream.shape(v), @initial_accumulator_value)
            init = TensorStream.cast(init_constant, dtype)
          end
          get_or_make_slot_with_initializer(v, init, v.shape, dtype, "accumulator", @name)
        end
      end

      def prepare
        learning_rate = call_if_callable(@learning_rate)
        @learning_rate_tensor = TensorStream.convert_to_tensor(learning_rate, name: "learning_rate")
      end

      def apply_dense(grad, var)
        acc = get_slot(var, "accumulator")
        _op(:apply_adagrad,
          var, acc, TensorStream.cast(@learning_rate_tensor, var.data_type),
          grad, use_locking: @use_locking)
      end
    end
  end
end
