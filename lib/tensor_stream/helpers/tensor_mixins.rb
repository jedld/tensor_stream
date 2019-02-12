module TensorStream
  module TensorMixins
    def +(other)
      TensorStream.check_data_types(self, other)
      _op(:add, self, other)
    end

    def [](index)
      _op(:index, self, index)
    end

    def *(other)
      TensorStream.check_data_types(self, other)
      _op(:mul, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def **(other)
      TensorStream.check_data_types(self, other)
      _op(:pow, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def /(other)
      TensorStream.check_data_types(self, other)
      _op(:div, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def -(other)
      TensorStream.check_data_types(self, other)
      _op(:sub, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def -@
      _op(:negate, self)
    end

    def %(other)
      TensorStream.mod(self, other)
    end

    def floor(name: nil)
      TensorStream.floor(self, name: name)
    end

    def ceil(name: nil)
      TensorStream.ceil(self, name: name)
    end

    def round(name: nil)
      TensorStream.round(self, name: name)
    end

    def log(name: nil)
      TensorStream.log(self, name: name)
    end

    def reshape(shape, name: nil)
      TensorStream.reshape(self, shape, name: name)
    end

    def zero?
      _op(:equal, self, TensorStream.constant(0, dtype: data_type, name: "equal/is_zero?"))
    end

    def ==(other)
      TensorStream.check_data_types(self, other)
      _op(:equal, self, other)
    end

    def <(other)
      TensorStream.check_data_types(self, other)
      _op(:less, self, other)
    end

    def !=(other)
      TensorStream.check_data_types(self, other)
      _op(:not_equal, self, other)
    end

    def >(other)
      TensorStream.check_data_types(self, other)
      _op(:greater, self, other)
    end

    def >=(other)
      TensorStream.check_data_types(self, other)
      _op(:greater_equal, self, other)
    end

    def <=(other)
      TensorStream.check_data_types(self, other)
      _op(:less_equal, self, other)
    end

    def and(other)
      TensorStream.check_data_types(self, other)
      _op(:logical_and, self, other)
    end

    def matmul(other)
      TensorStream.check_data_types(self, other)
      _op(:mat_mul, self, other)
    end

    def dot(other)
      TensorStream.check_data_types(self, other)
      _op(:mat_mul, self, other)
    end

    def cast(data_type = :float32, name: nil)
      TensorStream.cast(self, data_type, name: name)
    end

    def var(name: nil)
      TensorStream.variable(self, name: name)
    end

    ##
    # Apply a reduction to tensor
    def reduce(op_type = :+, axis: nil, keepdims: false, name: nil)
      reduce_op = case op_type.to_sym
                  when :+
                    :sum
                  when :*
                    :prod
                  when :mean
                    :mean
                  else
                    raise "unsupported reduce op type #{op_type} valid values are :+, :*, :prod, :mean"
      end
      raise "blocks are not supported for tensors" if block_given?

      TensorStream.reduce(reduce_op, self, axis, keepdims: keepdims, name: name)
    end
  end
end
