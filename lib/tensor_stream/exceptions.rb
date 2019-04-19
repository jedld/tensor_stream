module TensorStream
  class TensorStreamError < RuntimeError; end
  class KeyError < TensorStreamError; end
  class ValueError < TensorStreamError; end
  class TypeError < TensorStreamError; end
  class InvalidArgumentError < TensorStreamError; end
  class NotImplementedError < TensorStreamError; end
end
