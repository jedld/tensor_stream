require 'pry-byebug'
module TensorStream
  # various monkey patches to FixNum types
  module MonkeyPatch
    def self.included(klass)
      ops = if klass == Array
              {:+ => 'add', :- => 'sub', :* => 'mul'}
            else
              {:+ => 'add', :- => 'sub', :/ => 'div', :% => 'mod', :* => 'mul', :** => 'pow' }
            end

      ops.each do |m, name|
        klass.alias_method :"_tensor_stream_#{name}_orig", m
        klass.remove_method m
      end
    end

    def t(name = nil)
      TensorStream.convert_to_tensor(self, name: name)
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
        _tensor_stream_div_orig(other)
      end
    end

    def %(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self) % other
      else
        _tensor_stream_mod_orig(other)
      end
    end

    def **(other)
      if other.is_a?(TensorStream::Tensor)
        TensorStream.convert_to_tensor(self)**other
      else
        _tensor_stream_pow_orig(other)
      end
    end
  end
end