module TensorStream
  # various monkey patches to FixNum types
  module MonkeyPatch
    def self.included(klass)
      klass.alias_method :_tensor_stream_add_orig, :+
      klass.alias_method :_tensor_stream_sub_orig, :-
      klass.alias_method :_tensor_stream_mul_orig, :*
      klass.alias_method :_tensor_stream_div_orig, :/
      klass.alias_method :_tensor_stream_mod_orig, :%
      klass.remove_method :+
      klass.remove_method :-
      klass.remove_method :*
      klass.remove_method :/
      klass.remove_method :%
    end

    def +(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self) + other
      else
        _tensor_stream_add_orig(other)
      end
    end

    def -(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self) - other
      else
        _tensor_stream_sub_orig(other)
      end
    end

    def *(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self) * other
      else
        _tensor_stream_mul_orig(other)
      end
    end

    def /(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self) * other
      else
        _tensor_stream_mul_orig(other)
      end
    end

    def %(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self) % other
      else
        _tensor_stream_mod_orig(other)
      end
    end
  end
end