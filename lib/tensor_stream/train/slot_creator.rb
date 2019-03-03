module TensorStream
  module Train
    module SlotCreator
      include TensorStream::Utils

      ##
      # Helper function for creating a slot variable.
      def create_slot_var(_primary, val, scope, shape)
        slot = get_variable(scope, initializer: val, trainable: false, shape: shape,
                                                  validate_shape: val.shape && val.shape.known?)
        slot
      end

      ##
      # Create a slot initialized to the given value
      #
      # Args:
      #   primary: Variable - The primary 'Variable' or 'Tensor'
      #   val: Tensor - A `Tensor` specifying the initial value of the slot
      #   name: String - Name to use for the slot variable
      #   colocate_with_primary: Boolean - If true the slot is located
      #                                    on the same device as `primary`
      #
      # Returns: A `Variable` object
      def create_slot(primary, val, name, colocate_with_primary: true)
        TensorStream.variable_scope(nil, primary.op.name + "/" + name) do
          return create_slot_var(primary, val, "", nil) if colocate_with_primary

          TensorStream.colocate_with(primary) do
            return create_slot_var(primary, val, "", nil)
          end
        end
      end

      def create_slot_with_initializer(primary, initializer, shape, dtype, name, colocate_with_primary: true)
        prefix = primary.op.name
        TensorStream.variable_scope(nil, prefix + "/" + name) do
          create_slot_var(primary, initializer, "", shape.shape)
        end
      end

      ##
      # Create a slot initialized to 0 with same shape as the primary object.
      #
      # Args:
      #   primary: The pirmary variable or Tensor
      #   name: String - Name to use for the slot variable
      #   dtype: Symbol - Type of the slot variable
      #   colocate_with_primary: boolean - If true the slot is located on the same device as primary
      #
      # Returns:
      #   A `Variable` object
      def create_zeros_slot(primary, name, dtype: nil, colocate_with_primary: true)
        dtype = primary.data_type if dtype.nil?
        slot_shape = primary.shape
        slot_shape = if slot_shape.fully_defined?
          slot_shape.shape
        else
          TensorStream.shape(primary.initialized_value)
        end
        val = TensorStream.zeros(slot_shape, dtype: dtype)
        create_slot(primary, val, name,
          colocate_with_primary: colocate_with_primary)
      end
    end
  end
end
