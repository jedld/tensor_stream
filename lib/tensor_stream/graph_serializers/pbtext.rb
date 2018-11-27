module TensorStream
  # Parses pbtext files and loads it as a graph
  class Pbtext < TensorStream::Serializer
    include TensorStream::StringHelper
    include TensorStream::OpHelper

    def get_string(tensor_or_graph, session = nil, graph_keys = nil)
      graph = tensor_or_graph.is_a?(Tensor) ? tensor_or_graph.graph : tensor_or_graph
      @lines = []

      node_keys = graph_keys.nil? ? graph.node_keys : graph.node_keys.select { |k| graph_keys.include?(k) }

      node_keys.each do |k|
        node = if block_given?
                 yield graph, k
               else
                 graph.get_tensor_by_name(k)
               end

        @lines << "node {"
        @lines << "  name: #{node.name.to_json}"
        if node.is_a?(TensorStream::Operation)
          @lines << "  op: #{camelize(node.operation.to_s).to_json}"
          node.inputs.each do |input|
            next unless input

            @lines << "  input: #{input.name.to_json}"
          end
          # type
          pb_attr('T', "type: #{sym_to_protobuf_type(node.data_type)}")

          case node.operation.to_s
          when 'const'
            pb_attr('value', tensor_value(node))
          when 'variable_v2'
            pb_attr('shape', shape_buf(node, 'shape'))
          end
          process_options(node)
        end
        @lines << "}"
      end
      @lines << "versions {"
      @lines << "  producer: 26"
      @lines << "}"
      @lines.flatten.join("\n")
    end

    private

    def process_options(node)
      return if node.options.nil?
      node.options.each do |k, v|
        next if %w[name internal_name data_type].include?(k.to_s) || k.to_s.start_with?('__')
        @lines << "  attr {"
        @lines << "    key: \"#{k}\""
        @lines << "    value {"
        if v.is_a?(TrueClass) || v.is_a?(FalseClass)
          @lines << "      b: #{v}"
        elsif v.is_a?(Integer)
          @lines << "      i: #{v}"
        elsif v.is_a?(Float)
          @lines << "      f: #{v}"
        elsif v.is_a?(String)
          @lines << "      s: #{v}"
        elsif v.is_a?(Array)
          @lines << "      list {"
          v.each do |v_item|
            attr_value(v_item, 9)
          end
          @lines << "      }"
        end
        @lines << "    }"
        @lines << "  }"
      end
    end

    def attr_value(val, indent = 0)
      spaces = " " * indent
      case val.class
      when Integer
        @lines << "#{spaces}i: #{v_item}"
      when String
        @lines << "#{spaces}s: #{v_item}"
      when Float
        @lines << "#{spaces}f: #{v_item}"
      end
    end

    def pack_arr_float(float_arr)
      float_arr.flatten.pack('f*').bytes.map { |b| b.chr =~ /[^[:print:]]/ ? "\\#{sprintf("%o", b).rjust(3, '0')}" : b.chr }.join
    end

    def pack_arr_int(int_arr)
      int_arr.flatten.pack('l*').bytes.map { |b| b.chr =~ /[^[:print:]]/ ? "\\#{sprintf("%o", b).rjust(3, '0')}" : b.chr }.join
    end

    def shape_buf(tensor, shape_type = 'tensor_shape')
      arr = []
      arr << "  #{shape_type} {"
      if tensor.shape.shape
        tensor.shape.shape.each do |dim|
          arr << "    dim {"
          arr << "      size: #{dim}"
          arr << "    }"
        end
      end
      arr << "  }"
      arr
    end

    def tensor_value(tensor)
      arr = []
      arr << "tensor {"
      arr << "  dtype: #{sym_to_protobuf_type(tensor.data_type)}"

      arr += shape_buf(tensor)

      if tensor.rank > 0
        if TensorStream::Ops::FLOATING_POINT_TYPES.include?(tensor.data_type)
          packed = pack_arr_float(tensor.const_value)
          arr << "  tensor_content: \"#{packed}\""
        elsif TensorStream::Ops::INTEGER_TYPES.include?(tensor.data_type)
          packed = pack_arr_int(tensor.const_value)
          arr << "  tensor_content: \"#{packed}\""
        elsif tensor.data_type == :string
          tensor.const_value.each do |v|
            arr << "  string_val: #{v.to_json}"
          end
        else
          arr << "  tensor_content: #{tensor.const_value.flatten}"
        end
      else
        val_type = if TensorStream::Ops::INTEGER_TYPES.include?(tensor.data_type)
                     "int_val"
                   elsif TensorStream::Ops::FLOATING_POINT_TYPES.include?(tensor.data_type)
                     "float_val"
                   elsif tensor.data_type == :string
                     "string_val"
                   else
                     "val"
                   end
        arr << "  #{val_type}: #{tensor.const_value.to_json}"
      end
      arr << "}"
      arr
    end

    def sym_to_protobuf_type(type)
      case type
      when :int32, :int
        "DT_INT32"
      when :int16
        "DT_INT16"
      when :float, :float32
        "DT_FLOAT"
      when :float64
        "DT_FLOAT64"
      when :string
        "DT_STRING"
      else
        "UKNOWN"
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