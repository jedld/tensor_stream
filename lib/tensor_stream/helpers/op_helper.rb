module TensorStream
  # module that contains helper functions useful for ops
  module OpHelper
    def _op(code, t_a, t_b = nil, options = {})
      Operation.new(code.to_sym, t_a, t_b, options)
    end

    # same as op but with a marker that it was internal generated
    def i_op(code, t_a, t_b = nil, options = {})
      Operation.new(code.to_sym, t_a, t_b, options.merge(internal: true))
    end

    def cons(value, options = {})
      TensorStream.constant(value, options)
    end

    def i_cons(value, options = {})
      TensorStream.constant(value, options.merge(internal: true))
    end

    def shape_eval(input, output_type = :int32)
      return [] unless input.is_a?(Array)
      arr = []
      arr_ptr = input

      Kernel.loop do
        arr << (TensorStream::Ops::FLOATING_POINT_TYPES.include?(output_type) ? arr_ptr.size.to_f : arr_ptr.size)
        arr_ptr = arr_ptr[0]

        break unless arr_ptr.is_a?(Array)
      end

      arr
    end

    def dtype_eval(rank, value)
      dtype = Tensor.detect_type(value[0])

      rank += 1 if dtype == :array

      [dtype, rank, value[0], value.size]
    end

    def val_to_dtype(value)
      if value.is_a?(String)
        :string
      elsif value.is_a?(Float)
        :float32
      elsif value.is_a?(Integer)
        :int32
      elsif value.is_a?(Array)
        :array
      else
        :float32
      end
    end

    def fp_type?(type)
      TensorStream::Ops::FLOATING_POINT_TYPES.include?(type)
    end

    def format_source(trace)
      grad_source = trace.select { |c| c.to_s.include?(File.join('lib', 'tensor_stream', 'math_gradients')) }.first
      source = trace.reject { |c| c.to_s.include?(File.join('lib', 'tensor_stream')) }.first
      [grad_source, source].compact.join("\n")
    end
  end
end
