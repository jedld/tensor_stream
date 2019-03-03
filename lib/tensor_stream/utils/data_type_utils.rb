module TensorStream
  class DataTypeUtils
    def self.norm_dtype(dtype)
      dtype = dtype.to_sym
      case dtype
      when :int
        :int32
      when :float
        :float32
      else
        dtype
      end
    end
  end
end