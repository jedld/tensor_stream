module TensorStream
  class Serializer
    def serialize(filename, tensor, session = nil)
      File.write(filename, get_string(tensor, session))
    end

    def get_string(tensor, session = nil); end
  end
end