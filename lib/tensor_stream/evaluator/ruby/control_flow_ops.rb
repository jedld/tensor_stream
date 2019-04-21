module TensorStream
  module ControlFlowOps
    def self.included(klass)
      klass.class_eval do
        register_op :enter do |context, tensor, inputs|
        end
      end
    end
  end
end