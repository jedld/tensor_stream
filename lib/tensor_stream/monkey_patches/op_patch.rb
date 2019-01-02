module TensorStream
  module OpPatch
    def self.included(klass)
      ops = if klass == Array
              {:+ => 'add', :- => 'sub', :* => 'mul'}
            else
              {:+ => 'add', :- => 'sub', :/ => 'div', :% => 'mod', :* => 'mul', :** => 'pow' }
            end

      ops.each do |m, name|
        klass.send(:alias_method, :"_tensor_stream_#{name}_orig", m)
        klass.send(:remove_method, m)
      end
    end

    def +(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self, dtype: other.data_type) + other
      else
        _tensor_stream_add_orig(other)
      end
    end

    def -(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self, dtype: other.data_type) - other
      else
        _tensor_stream_sub_orig(other)
      end
    end

    def *(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self, dtype: other.data_type) * other
      else
        _tensor_stream_mul_orig(other)
      end
    end

    def /(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self, dtype: other.data_type) * other
      else
        _tensor_stream_div_orig(other)
      end
    end

    def %(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self, dtype: other.data_type) % other
      else
        _tensor_stream_mod_orig(other)
      end
    end

    def **(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self, dtype: other.data_type)**other
      else
        _tensor_stream_pow_orig(other)
      end
    end
  end
end

Integer.include TensorStream::OpPatch
Float.include TensorStream::OpPatch
Array.include TensorStream::OpPatch