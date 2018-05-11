module TensorStream
  module OpHelper
    def op(code, a, b = nil, options = {})
      Operation.new(code.to_sym, a, b, options)
    end

    # same as op but with a marker that it was internal generated
    def i_op(code, a, b = nil, options = {})
      Operation.new(code.to_sym, a, b, options.merge(internal: true))
    end

    def cons(value, options = {})
      TensorStream.constant(value, options)
    end

    def i_cons(value, options = {})
      TensorStream.constant(value, options.merge(internal: true))
    end

    def shape_eval(input)
      return [] unless input.kind_of?(Array)
      arr = []
      arr_ptr = input

      Kernel.loop do
        arr << arr_ptr.size
        arr_ptr = arr_ptr[0]

        break unless arr_ptr.is_a?(Array)
      end

      arr
    end

  def dtype_eval(dtype, rank, value)
    dtype = Tensor.detect_type(value[0])
    rank+=1 if dtype == :array

    [dtype, rank, value[0], value.size]
  end

  def val_to_dtype(value, rank = 0)
    dtype = if value.is_a?(String)
      :string
    elsif value.is_a?(Float)
      :float32
    elsif value.is_a?(Integer)
      :int32
    elsif value.is_a?(Array)
      rank += 1
      :array
    else
      :float32
    end
    dtype
  end
  end
end
