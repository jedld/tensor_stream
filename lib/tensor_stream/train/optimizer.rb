module TensorStream
  module Train
    # Base class for an optimizer
    # This is a straight up port from the python version
    class Optimizer
      include SlotCreator

      attr_reader :name

      def initialize(name:, use_locking:)
        @name = name
        @use_locking = use_locking
        raise TensorStream::ValueError, "Must specify the optimizer name" unless @name

        @slots = {}
        @non_slots = {}
      end

      def minimize(loss, var_list: nil, grad_loss: nil, global_step: nil, name: nil)
        grads_and_vars = compute_gradients(loss, var_list: var_list, grad_loss: grad_loss)
        apply_gradients(grads_and_vars, global_step: global_step, name: name)
      end

      ##
      # Apply gradients to variables.
      # This is the second part of minimize(). It returns an Operation that applies gradients.
      def apply_gradients(grads_and_vars, global_step: nil, name: nil)
        varlist = grads_and_vars.map { |_grad, var| var }
        create_slots(varlist)
        TensorStream.name_scope(name, default: @name) do
          prepare
          apply_ops = grads_and_vars.map { |grad, var|
            TensorStream.name_scope("update_" + var.op.name) do
              apply_dense(grad, var)
            end
          }

          if global_step.nil?
            finish(apply_ops, name)
          else
            TensorStream.control_dependencies([finish(apply_ops, "update")]) do
              global_step.assign_add(1)
            end
          end
        end
      end

      ##
      # Compute gradients of loss for the variables in var_list.
      #
      # This is the first part of minimize(). It returns a list of (gradient, variable) pairs where "gradient" is the gradient for "variable".
      def compute_gradients(loss, var_list: nil, grad_loss: nil)
        trainable_vars = if var_list
          raise "var_list must be an array" unless var_list.is_a?(Array)

          var_list.each_with_index { |var, index| raise "var #{index} not a Variable" unless var.is_a?(Variable) }

          var_list
        else
          loss.graph.get_collection(TensorStream::GraphKeys::TRAINABLE_VARIABLES)
        end
        all_grads = grad_loss || TensorStream.gradients(loss, trainable_vars)
        trainable_vars.each_with_index.collect do |var, index|
          [all_grads[index], var]
        end
      end

      def get_slot(var, name)
        named_slots = @slots.fetch(name, nil)
        return nil if named_slots.nil?

        named_slots.fetch(var_key(var), nil)
      end

      def get_slot_names
        @slots.keys.sort
      end

      protected

      def finish(update_ops, name_scope)
        TensorStream.group(update_ops, name: name_scope)
      end

      def create_slots(var_list)
        # no implementation
      end

      def prepare
        # no implementation
      end

      def apply_dense(_grad, _var)
        raise TensorStream::NotImplementedError, "not implemented"
      end

      ##
      # Find or create a slot initialized with 0.0.
      #
      # Args:
      #   var: Variable - A Variable object
      #   slot_name: string - Name for the slot
      #   op_name: string - Name to use when scoping the Variable that needs to be created
      def zeros_slot(var, slot_name, op_name)
        named_slots = slot_dict(slot_name)
        unless named_slots.key?(var_key(var))
          named_slots[var_key(var)] = create_zeros_slot(var, op_name)
        end
        named_slots[var_key(var)]
      end

      ##
      # Returns a dict for caching slots created under the given name.
      #
      # Args:
      # slot_name string Name for the slot
      #
      # Returns: A dict that maps primary 'Variable' objects to the slot created
      def slot_dict(slot_name)
        named_slots = @slots.fetch(slot_name, nil)
        if named_slots.nil?
          named_slots = {}
          @slots[slot_name] = named_slots
        end
        named_slots
      end

      def var_key(var)
        [var.op.graph, var.op.name]
      end

      def get_non_slot_variable(name, graph: nil)
        non_slot = @non_slots.fetch([name, graph], nil)
        non_slot
      end

      def call_if_callable(param)
        param.is_a?(Proc) ? param.call : param
      end

      def create_non_slot_variable(initial_value, name, colocate_with)
        graph = colocate_with.graph

        key = [name, graph]
        v = @non_slots.fetch(key, nil)
        if v.nil?
          v = TensorStream.variable(initial_value, name: name, trainable: false)
          @non_slots[key] = v
        end
        v
      end

      ##
      # Find or create a slot for a variable, using an Initializer.
      def get_or_make_slot_with_initializer(var, initializer, shape, dtype, slot_name, op_name)
        named_slots = slot_dict(slot_name)
        unless named_slots.key?(var_key(var))
          new_slot_variable = create_slot_with_initializer(var, initializer, shape, dtype, op_name)
          named_slots[var_key(var)] = new_slot_variable
        end
        named_slots[var_key(var)]
      end
    end
  end
end
