module TensorStream
  class Serializer
    def serialize(filename, tensor, session = nil, graph_keys = nil)
      File.write(filename, get_string(tensor, session, graph_keys = nil))
    end

    def get_string(tensor, session = nil)
    end
  end
end
