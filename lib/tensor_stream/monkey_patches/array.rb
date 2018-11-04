class Array
  include TensorStream::MonkeyPatch

  def /(other)
    TensorStream.convert_to_tensor(self) * other
  end

  def %(other)
    TensorStream.convert_to_tensor(self) % other
  end

  def **(other)
    TensorStream.convert_to_tensor(self)**other
  end
end