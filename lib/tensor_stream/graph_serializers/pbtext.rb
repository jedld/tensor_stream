module TensorStream
  class Pbtext
    def initialize
    end

    def serialize(session, filename, tensor)
    end

    def get_string(graph)
      @lines = []
      graph.nodes.each do |k, node|
        @lines << "node {"
        @lines << "  name: #{node.name.to_json}"
        if node.is_a?(TensorStream::Operation)
          @lines << "  op: #{node.operation.to_json}"
          node.items.each do |input|
            next unless input
            @lines << "  input: #{input.name.to_json}"
          end
          # type
          pb_attr('T', sym_to_protobuf_type(node.data_type))
        elsif node.is_a?(TensorStream::Tensor) && node.is_const
          @lines << "  op: \"Const\""
          # type
          pb_attr('T', sym_to_protobuf_type(node.data_type))
          pb_attr('value', tensor_value(node))
        end
        @lines << "}"
      end
      @lines.join("\n")
    end

    private

    def tensor_value(tensor)
      arr = []
      arr << "tensor {"
      arr << "  dtype: #{sym_to_protobuf_type(tensor.data_type)}"
      arr << "  float_val: #{tensor.value}"
      arr << "}"
      arr
    end

    def sym_to_protobuf_type(type)
      case type
      when :int32
        "DT_INT32"
      when :float, :float32
        "DT_FLOAT"
      else
        "DT_UNKNOWN"
      end
    end

    def pb_attr(key, value)
      @lines << "  attr {"
      @lines << "    key: \"#{key}\""
      @lines << "    value {"
      if value.is_a?(Array)
        value.each do |v|
          @lines << "      #{v}"
        end
      else
        @lines << "      #{value}"
      end
      @lines << "    }"
      @lines << "  }"
    end
  end

end