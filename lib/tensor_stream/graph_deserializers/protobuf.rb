require 'yaml'

module TensorStream
  # A .pb graph deserializer
  class Protobuf
    def initialize
    end

    def load_from_string(buffer)
      evaluate_lines(buffer.split("\n").map(&:strip))
    end

    ##
    # parsers a protobuf file and spits out
    # a ruby hash
    def load(pbfile)
      f = File.new(pbfile, 'r')
      lines = []
      while !f.eof? && (str = f.readline.strip)
        lines << str
      end
      evaluate_lines(lines)
    end

    protected

    def evaluate_lines(lines = [])
      block = []
      node = {}
      node_attr = {}
      dim = []
      state = :top

      lines.each do |str|
        case(state)
        when :top
          node['type'] = parse_node_name(str)
          state = :node_context
          next
        when :node_context
          if str == 'attr {'
            state = :attr_context
            node_attr = {}
            node['attributes']||=[]
            node['attributes'] << node_attr
            next
          elsif str == '}'
            state = :top
            block << node
            node = {}
            next
          else
            key, value = str.split(':')
            if key == 'input'
              node['input']||=[]
              node['input'] << process_value(value.strip)
            else
              node[key] = process_value(value.strip)
            end
          end
        when :attr_context
          if str == 'value {'
            state = :value_context
            node_attr['value'] = {}
            next
          elsif str == '}'
            state = :node_context
            next
          else
            key, value = str.split(':')
            node_attr[key] = process_value(value.strip)
          end
        when :value_context
          if str == 'list {'
            state = :list_context
            next
          elsif str == 'shape {'
            state = :shape_context
            next
          elsif str == 'tensor {'
            state = :tensor_context
            node_attr['value']['tensor'] = {}
            next
          elsif str == '}'
            state = :attr_context
            next
          else
            key, value = str.split(':')
            if key == 'dtype'
              node_attr['value']['dtype'] = value.strip
            elsif key === 'type'
              node_attr['value']['type'] = value.strip
            else
              node_attr['value'][key] = process_value(value.strip)
            end
          end
        when :list_context
          if str == '}'
            state = :value_context
            next
          end
        when :tensor_context
          if str == 'tensor_shape {'
            state = :tensor_shape_context
            node_attr['value']['tensor']['shape'] = []
            next
          elsif str == '}'
            state = :value_context
            next
          else
            key, value = str.split(':')
            node_attr['value']['tensor'][key] = process_value(value.strip)
          end
        when :tensor_shape_context
          if str == 'dim {'
            state = :tensor_shape_dim_context
            next
          elsif str == '}'
            state = :tensor_context
            next
          end
        when :shape_context
          if str == '}'
            state = :value_context
            next
          end
        when :tensor_shape_dim_context
          if str == '}'
            state = :tensor_shape_context
            next
          else
            key, value = str.split(':')
            node_attr['value']['tensor']['shape'] << value.strip.to_i
          end
        end
      end

      block
    end

    def parse_node_name(str)
      name = str.split(' ')[0]
    end

    def process_value(value)
      if value.start_with?('"')
        unescape(value.gsub!(/\A"|"\Z/, ''))
      else
        unescape(value)
      end
    end

    UNESCAPES = {
      'a' => "\x07", 'b' => "\x08", 't' => "\x09",
      'n' => "\x0a", 'v' => "\x0b", 'f' => "\x0c",
      'r' => "\x0d", 'e' => "\x1b", "\\\\" => "\x5c",
      "\"" => "\x22", "'" => "\x27"
    }

    def unescape(str)
      # Escape all the things
      str.gsub(/\\(?:([#{UNESCAPES.keys.join}])|u([\da-fA-F]{4}))|\\0?x([\da-fA-F]{2})/) {
        if $1
          if $1 == '\\' then '\\' else UNESCAPES[$1] end
        elsif $2 # escape \u0000 unicode
          ["#$2".hex].pack('U*')
        elsif $3 # escape \0xff or \xff
          [$3].pack('H2')
        end
      }
    end
  end
end