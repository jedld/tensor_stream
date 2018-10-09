require 'base64'

module TensorStream
  # Utility class to handle data type serialization
  class Packer
    def self.pack(value, data_type)
      value = value.is_a?(Array) ? value.flatten : [value]
      byte_value = case data_type
                   when :float64
                     value.pack('d*')
                   when :float32, :float16, :float
                     value.pack('f*')
                   when :uint32
                     value.pack('L*')
                   when :int32, :int
                     value.pack('l*')
                   when :int64
                     value.pack('q*')
                   when :uint64
                     value.pack('Q*')
                   when :uint8
                     value.pack('C*')
                   when :boolean
                     value.map { |v| v ? 1 : 0 }.pack('C*')
                   end

      byte_value
    end

    def self.pack_to_str(value, data_type)
      pack(value, data_type).bytes.map { |b| b.chr =~ /[^[:print:]]/ ? "\\#{sprintf("%o", b).rjust(3, '0')}" : b.chr }.join
    end

    def self.unpack_from_str(content, data_type)
      unpacked = eval(%Q("#{content}"))
      unpack(unpacked, data_type)
    end

    def self.unpack(unpacked, data_type)
      case data_type
      when :float32, :float, :float16
        unpacked.unpack('f*')
      when :float64
        unpacked.unpack('d*')
      when :int32, :int
        unpacked.unpack('L*')
      when :uint32
        unpacked.unpack('l*')
      when :int64
        unpacked.unpack('q*')
      when :uint64
        unpacked.unpack('Q*')
      when :uint8
        unpacked.unpack('C*')
      when :boolean
        unpacked.unpack('C*').map { |v| v == 1 }
      end
    end
  end
end