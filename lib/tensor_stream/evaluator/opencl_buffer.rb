module TensorStream
  class OpenCLBuffer < Buffer
    attr_accessor :data_type, :shape, :buffer, :cl_buffer, :op

    def initialize(data_type: , shape:, buffer:, cl_buffer:, op: nil, name: nil)
      @data_type = data_type
      @shape = shape
      @buffer = buffer
      @cl_buffer = cl_buffer
      @name = name
      @op = op
    end

    def to_ruby
      return buffer[0] if shape.empty?
      return [] if buffer.empty?
      buffer.reshape(*shape.reverse).to_a
    end
  end
end