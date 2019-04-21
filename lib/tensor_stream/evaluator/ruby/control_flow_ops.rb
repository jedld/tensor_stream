module TensorStream
  module ControlFlowOps
    def self.included(klass)
      klass.class_eval do
        register_op :enter do |context, tensor, inputs|
        end

        register_op :merge do |context, tensor, inputs|
        end

        register_op :loop_cond do |context, tensor, inputs|
        end

        register_op :switch do |context, tensor, inputs|
        end

        register_op :next_iteration do |context, tensor, inputs|
        end

        register_op :exit do |context, tensor, inputs|
        end
      end
    end
  end
end