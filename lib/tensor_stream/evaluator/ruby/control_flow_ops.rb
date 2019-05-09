module TensorStream
  module ControlFlowOps
    def self.included(klass)
      klass.class_eval do
        register_op :enter, noop: true do |context, tensor, inputs|
          binding.pry
          context[:global][:while_context] ||= []
          context[:global][:while_context] << { start: inputs, next_iteration: inputs[0] }
          inputs[0]
        end

        register_op :merge, noop: true do |context, tensor, inputs|
          result = nil
          tensor.inputs.each_with_index do |t, index|
            val = global_eval(tensor, t, context)
            if val
              result = [val, index]
              break
            end
          end
          result
        end

        register_op :loop_cond, noop: true do |context, tensor, inputs|
          while_context = context[:global][:while_context].last
          while_context[:cond] = tensor.inputs[0]

          inputs[0]
        end

        register_op :switch, noop: true do |context, tensor, inputs|
          tensor.inputs.map do |x, cond|
            cond_value = global_eval(tensor, cond, context)
            cond_value ? global_eval(tensor, x, context) : nil
          end
        end

        register_op :next_iteration, noop: true do |context, tensor, inputs|
          while_context = context[:global][:while_context].last
          while_context[:next_iteration] = tensor.inputs[0]
          inputs[0]
        end

        register_op :exit do |context, tensor, inputs|
          binding.pry
          # while_context = context[:global][:while_context].last
          # while_context[:loop_values]
        end
      end
    end
  end
end