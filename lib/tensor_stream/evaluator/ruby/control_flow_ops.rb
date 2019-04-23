module TensorStream
  module ControlFlowOps
    def self.included(klass)
      klass.class_eval do
        register_op :enter do |context, tensor, inputs|
          context[:global][:while_context] ||= []
          context[:global][:while_context] << { start: inputs }
          inputs[0]
        end

        register_op :merge do |context, tensor, inputs|
          result = nil
          inputs.each_with_index do |val, index|
            if val
              result = [val, index]
              break
            end
          end

          result
        end

        register_op :loop_cond do |context, tensor, inputs|
          while_context = context[:global][:while_context].last
          while_context[:cond] = tensor.inputs[0]
          inputs[0]
        end

        register_op :switch do |context, tensor, inputs|
          inputs.map do |x, cond|
            cond ? x : nil
          end
        end

        register_op :next_iteration do |context, tensor, inputs|
          while_context = context[:global][:while_context].last
          while_context[:next_iteration] = tensor.inputs[0]
          binding.pry
          inputs[0]
        end

        register_op :exit do |context, tensor, inputs|
          binding.pry
          inputs[0]
        end
      end
    end
  end
end