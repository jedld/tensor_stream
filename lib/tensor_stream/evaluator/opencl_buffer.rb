module TensorStream
  class OpenCLBuffer < Buffer
    include ArrayOpsHelper

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
      if shape.empty?
        return buffer[0] != 0 if data_type == :boolean
        return buffer[0]
      end
      return [] if buffer.empty?

      result = buffer.reshape(*shape.reverse).to_a
      if data_type == :boolean
        result = process_function_op(result, ->(a, _b) { a != 0 })
      end
      result
    end
  end
end