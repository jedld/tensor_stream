class Integer
  include TensorStream::MonkeyPatch

  def self.placeholder(name: nil, width: 32, shape: nil)
    raise "invalid width passed #{width}" unless [16, 32, 64].include?(width)

    data_type = :"int#{width}"
    TensorStream.placeholder(data_type, name: name, shape: shape)
  end
end
