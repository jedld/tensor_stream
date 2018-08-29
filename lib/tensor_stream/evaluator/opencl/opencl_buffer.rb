module TensorStream
  # Buffer used by the OpenCL evaluator
  class OpenCLBuffer < Buffer
    include ArrayOpsHelper

    attr_accessor :shape, :buffer, :cl_buffer, :op

    def initialize(data_type:, shape:, buffer:, cl_buffer:, op: nil, name: nil)
      @data_type = data_type
      @shape = shape
      @buffer = buffer
      @cl_buffer = cl_buffer
      @name = name
      @op = op
    end

    def to_ruby
      return [] if buffer.empty?

      if dirty
        op.command_queue.enqueue_read_buffer(cl_buffer, buffer, event_wait_list: [op].compact)
        op.command_queue.finish
        self.dirty = false
      end

      if shape.empty?
        return buffer[0] != 0 if data_type == :boolean
        return buffer[0]
      end

      result = buffer.reshape(*shape.map(&:to_i).reverse).to_a
      data_type == :boolean ? process_function_op(result, ->(a, _b) { a != 0 }) : result
    end
  end
end
