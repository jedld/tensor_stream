module TensorStream
  class Pbtext < TensorStream::Serializer
    include TensorStream::StringHelper

    def get_string(tensor_or_graph, session = nil)
      graph = tensor_or_graph.is_a?(Tensor) ? tensor_or_graph.graph : tensor_or_graph
      @lines = []
      graph.nodes.each do |k, node|
        @lines << "node {"
        @lines << "  name: #{node.name.to_json}"
        if node.is_a?(TensorStream::Operation)
          @lines << "  op: #{camelize(node.operation.to_s).to_json}"
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

    def pack_arr_float(float_arr)
      float_arr.flatten.pack('f*').bytes.map { |b| b.chr =~ /[^[:print:]]/ ? "\\#{sprintf("%o", b).rjust(3, '0')}" : b.chr  }.join
    end
  
    def tensor_value(tensor)
      arr = []
      arr << "tensor {"
      arr << "  dtype: #{sym_to_protobuf_type(tensor.data_type)}"

      arr << "  tensor_shape {"
      tensor.shape.shape.each do |dim|
        arr << "    dim {"
        arr << "      size: #{dim}"
        arr << "    }"
      end if tensor.shape.shape
      arr << "  }"

      if tensor.rank > 0
        if TensorStream::Ops::FLOATING_POINT_TYPES.include?(tensor.data_type)
          packed = pack_arr_float(tensor.value)
          arr << "  tensor_content: \"#{packed}\""
        else
          arr << "  tensor_content: #{tensor.value.flatten}"
        end
      else
        val_type = if tensor.data_type == :int32
          "int_val"
        else
          "float_val"
        end
        arr << "  #{val_type}: #{tensor.value}"
      end
      arr << "}"
      arr
    end

    def sym_to_protobuf_type(type)
      case type
      when :int32, :int
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