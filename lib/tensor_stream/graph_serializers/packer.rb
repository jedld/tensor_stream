module TensorStream
  # Utility class to handle data type serialization
  class Packer
    def self.pack(value, data_type)
      value = value.flatten
      byte_value = case data_type
                   when :float64
                     value.pack('d*').bytes
                   when :float32, :float16, :float
                     value.pack('f*').bytes
                   when :uint32
                     value.pack('L*').bytes
                   when :int32, :int
                     value.pack('l*').bytes
                   when :int64
                     value.pack('q*').bytes
                   when :uint64
                     value.pack('Q*').bytes
                   when :uint8
                     value.pack('C*').bytes
                   when :boolean
                     value.map { |v| v ? 1 : 0 }.pack('C*').bytes
                   end

      byte_value.map { |b| b.chr =~ /[^[:print:]]/ ? "\\#{sprintf("%o", b).rjust(3, '0')}" : b.chr }.join
    end

    def self.unpack(content, data_type)
      unpacked = eval(%Q("#{content}"))
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