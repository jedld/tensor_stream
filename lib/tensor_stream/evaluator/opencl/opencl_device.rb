module TensorStream
  class OpenclDevice < TensorStream::Device
    attr_accessor :native_device
  end
end