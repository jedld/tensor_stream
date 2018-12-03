module TensorStream
  module TensorMixins
    def +(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:add, self, other)
    end

    def [](index)
      _op(:index, self, index)
    end

    def *(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:mul, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def **(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:pow, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def /(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:div, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def -(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:sub, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def -@
      _op(:negate, self)
    end

    def %(other)
      TensorStream.mod(self, other)
    end

    def floor
      TensorStream.floor(self)
    end

    def ceil
      TensorStream.ceil(self)
    end

    def zero?
      _op(:equal, self, TensorStream.constant(0, dtype: data_type, name: 'equal/is_zero?'))
    end

    def ==(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:equal, self, other)
    end

    def <(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:less, self, other)
    end

    def !=(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:not_equal, self, other)
    end

    def >(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:greater, self, other)
    end

    def >=(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:greater_equal, self, other)
    end

    def <=(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:less_equal, self, other)
    end

    def and(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:logical_and, self, other)
    end

    def matmul(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:mat_mul, self, other)
    end

    def dot(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:mat_mul, self, other)
    end

  end
end