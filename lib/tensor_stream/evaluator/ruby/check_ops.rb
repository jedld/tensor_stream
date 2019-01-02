module TensorStream
  module CheckOps
    def CheckOps.included(klass)
      klass.class_eval do
        register_op :assert_equal do |context, tensor, inputs|
          result = call_vector_op(tensor, :equal, inputs[0], inputs[1], context) { |t, u| t == u }

          result = result.is_a?(Array) ? result.flatten.uniq : [result]
          prefix = tensor.options[:message] || ""
          raise TensorStream::InvalidArgumentError, "#{prefix} #{tensor.inputs[0].name} != #{tensor.inputs[1].name}" if result != [true]

          nil
        end
      end
    end
  end
end