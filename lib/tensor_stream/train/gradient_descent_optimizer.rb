module TensorStream
  module Train
    # High Level implementation of the gradient descent algorithm
    class GradientDescentOptimizer
      include TensorStream::OpHelper

      attr_accessor :learning_rate

      def initialize(learning_rate, _options = {})
        @learning_rate = learning_rate
      end

      def minimize(loss, var_list: nil, grad_loss: nil, global_step: nil)
        grads_and_vars = compute_gradients(loss, var_list: var_list, grad_loss: grad_loss)
        apply_gradients(grads_and_vars, global_step: global_step)
      end

      ##
      # Apply gradients to variables.
      # This is the second part of minimize(). It returns an Operation that applies gradients.
      def apply_gradients(grads_and_vars, global_step: nil)
        apply_ops = grads_and_vars.map do |grad, var|
          i_op(:apply_gradient_descent, var, TensorStream.cast(@learning_rate, grad.data_type), grad)
        end

        if global_step.nil?
          apply_ops
        else
          apply_ops + [global_step.assign_add(1)]
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
    end
  end
end
