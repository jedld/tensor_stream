require "tensor_stream/evaluator/operation_helpers/array_ops_helper"
module TensorStream
  ##
  # Convenience class for guessing the shape of a tensor
  #
  class InferShape
    extend TensorStream::ArrayOpsHelper
    extend TensorStream::OpHelper

    def self.infer_shape(tensor)
      case tensor.operation
      when :assign
        possible_shape = if tensor.inputs[0]&.shape&.shape
          tensor.inputs[0].shape.shape
        else
          tensor.inputs[1].shape.shape
        end

        possible_shape
      when :const
        shape_eval(tensor.options[:value])
      when :variable_v2
        tensor.shape ? tensor.shape.shape : nil
      when :placeholder
        return nil if tensor.inputs[0].nil?
        return tensor.inputs[0].shape.shape if tensor.inputs.size == 1

        TensorShape.infer_shape(tensor.inputs[0].shape.shape, tensor.inputs[1].shape.shape) if tensor.inputs.size == 2 && tensor.inputs[0] && tensor.inputs[1]
      when :case, :case_grad
        tensor.inputs[2]&.shape&.shape
      when :const
        shape_eval(tensor.options[:value])
      when :variable_v2
        tensor.shape ? tensor.shape.shape : nil
      when :assign
        possible_shape = if tensor.inputs[0]&.shape&.shape
          tensor.inputs[0].shape.shape
        else
          tensor.inputs[1].shape.shape
        end

        possible_shape
      when :index
        return nil unless tensor.inputs[0].is_a?(Tensor)
        return nil unless tensor.inputs[0].const_value

        input_shape = tensor.inputs[0].shape
        return nil unless input_shape.known?

        s = input_shape.shape.dup
        s.shift
        s
      when :arg_min, :argmax, :argmin
        return nil unless tensor.inputs[0].shape.known?
        return nil if tensor.inputs[1] && tensor.inputs[1].const_value.nil?

        axis = tensor.inputs[1].nil? ? 0 : tensor.inputs[1].const_value
        new_shape = tensor.inputs[0].shape.shape
        new_shape.each_with_index.collect { |shape, index|
          next nil if index == axis

          shape
        }.compact
      when :mean, :prod, :sum, :arg_max
        return [] if tensor.inputs[1].nil?
        return nil if tensor.inputs[0].nil?
        return nil unless tensor.inputs[0].shape.known?

        input_shape = tensor.inputs[0].shape.shape
        rank = input_shape.size

        axis = tensor.inputs[1].const_value
        return nil if axis.nil?

        axis = [axis] unless axis.is_a?(Array)
        axis = axis.map { |a| a < 0 ? rank - a.abs : a }

        input_shape.each_with_index.map { |item, index|
          if axis.include?(index)
            next 1 if tensor.options[:keepdims]

            next nil
          end
          item
        }.compact
      when :flow_group
        []
      when :zeros, :ones, :fill, :random_standard_normal, :random_uniform, :truncated_normal
        a_shape = tensor.inputs[0] ? tensor.inputs[0].const_value : tensor.options[:shape]
        return nil if a_shape.nil?

        a_shape.is_a?(Array) ? a_shape : [a_shape]
      when :zeros_like, :ones_like
        tensor.inputs[0].shape.shape
      when :shape
        tensor.inputs[0].shape.shape ? [tensor.inputs[0].shape.shape.size] : nil
      when :pad
        return nil unless tensor.inputs[0].shape.known?
        return nil unless tensor.inputs[1].const_value

        size = tensor.inputs[0].shape.shape.reduce(:*) || 1
        dummy_tensor_for_shape = TensorShape.reshape(Array.new(size), tensor.inputs[0].shape)
        shape_eval(arr_pad(dummy_tensor_for_shape, tensor.inputs[1].const_value))
      when :transpose
        return nil unless shape_full_specified(tensor.inputs[0])
        return nil if tensor.inputs[1].is_a?(Tensor)

        rank = tensor.inputs[0].shape.shape.size
        perm = tensor.inputs[1] || (0...rank).to_a.reverse
        perm.map { |p| tensor.inputs[0].shape.shape[p] }
      when :stack
        return nil unless shape_full_specified(tensor.inputs[0])

        axis = tensor.options[:axis] || 0
        new_shape = [tensor.inputs.size]
        tensor.inputs[0].shape.shape.inject(new_shape) { |ns, i| ns << i }
        rank = tensor.inputs[0].shape.shape.size + 1
        axis = rank + axis if axis < 0
        rotated_shape = Array.new(axis + 1) { new_shape.shift }
        rotated_shape.rotate! + new_shape
      when :concat
        return nil if tensor.inputs[0].const_value.nil?

        axis = tensor.inputs[0].const_value # get axis

        axis_size = 0

        tensor.inputs[1..tensor.inputs.size].each do |input_item|
          return nil if input_item.shape.shape.nil?
          return nil if input_item.shape.shape[axis].nil?

          axis_size += input_item.shape.shape[axis]
        end

        new_shape = tensor.inputs[1].shape.shape.dup
        new_shape[axis] = axis_size
        new_shape
      when :slice, :squeeze
        nil
      when :broadcast_gradient_args
        nil
      when :no_op
        nil
      when :softmax_cross_entropy_with_logits_v2, :sparse_softmax_cross_entropy_with_logits
        nil
      when :decode_png, :flow_dynamic_stitch, :dynamic_stitch, :gather
        nil
      when :eye
        return [tensor.inputs[0].const_value, tensor.inputs[1].const_value] if tensor.inputs[0].const_value && tensor.inputs[1].const_value

        nil
      when :unstack
        return nil unless tensor.inputs[0].shape.known?

        new_shape = tensor.inputs[0].shape.shape.dup
        rank = new_shape.size - 1
        axis = tensor.options[:axis] || 0
        axis = rank + axis if axis < 0
        rotated_shape = Array.new(axis + 1) { new_shape.shift }
        rotated_shape.rotate!(-1) + new_shape
      when :conv2d
        return nil unless tensor.inputs[0].shape.known?
        return nil unless tensor.inputs[1].shape.known?

        new_shape = tensor.inputs[0].shape.shape.dup
        new_shape[3] = tensor.inputs[1].shape.shape[3]

        # account for stride and padding options
        strides = tensor.options[:strides]

        case tensor.options[:padding]
        when "SAME"
          new_shape[1] /= strides[1]
          new_shape[2] /= strides[2]
        when "VALID"
          new_shape[1] = (new_shape[1] - tensor.inputs[1].shape.shape[0]) / strides[1] + 1
          new_shape[2] = (new_shape[2] - tensor.inputs[1].shape.shape[1]) / strides[2] + 1
        else
          raise TensorStream::ValueError, "Invalid padding option only 'SAME', 'VALID' accepted"
        end

        new_shape
      when :conv2d_backprop_input
        return nil unless tensor.inputs[0].const_value

        tensor.inputs[0].const_value
      else
        TensorStream::OpMaker.infer_shape(self, tensor)
      end
    end

    def self._infer_reduction_op_shape(tensor)
      return [] if tensor.inputs[1].nil?
      return nil if tensor.inputs[0].nil?
      return nil unless tensor.inputs[0].shape.known?

      input_shape = tensor.inputs[0].shape.shape
      rank = input_shape.size

      axis = tensor.inputs[1].const_value
      return nil if axis.nil?

      axis = [axis] unless axis.is_a?(Array)
      axis = axis.map { |a| a < 0 ? rank - a.abs : a }

      input_shape.each_with_index.map { |item, index|
        if axis.include?(index)
          next 1 if tensor.options[:keepdims]

          next nil
        end
        item
      }.compact
    end
  end
end
