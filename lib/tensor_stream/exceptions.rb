module TensorStream
  class TensorStreamError < RuntimeError; end
  class KeyError < TensorStreamError; end
  class ValueError < TensorStreamError; end
  class InvalidArgumentError < TensorStreamError; end
end