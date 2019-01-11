# require 'pry-byebug'
module TensorStream
  # various monkey patches to FixNum types
  module MonkeyPatch
    def shape
      TensorStream.shape_eval(self)
    end

    def t(name = nil, dtype: nil)
      TensorStream.convert_to_tensor(self, name: name, dtype: dtype)
    end
  end
end
