require "ostruct"

module TensorStream
  # Base class that defines a tensor like interface
  class Tensor
    include OpHelper
    extend OpHelper
    include TensorMixins

    attr_reader :graph, :value
    attr_accessor :name, :data_type, :shape, :rank, :native_buffer, :is_const,
      :internal, :source, :given_name, :outputs, :op

    def inspect
    end

    def internal?
      !!@internal
    end

    def dtype
      @data_type
    end

    def consumers
      op.consumers
    end

    def self.reset_counters
      @const_counter = 0
      @var_counter = 0
      @placeholder_counter = 0
    end

    def device
      @op.device
    end

    def collect(&block)
      @value.collect(&block)
    end

    def to_s
      @name
    end

    def eval(options = {})
      Session.default_session.run(self, options)
    end

    def to_h
      {
      }
    end

    def to_i
      @value
    end

    def to_a
      @value
    end

    def to_f
      @value
    end

    def first
      _op(:index, self, 0)
    end

    def to_math(name_only = false, max_depth = 99, _unused = 0)
      return @name if max_depth.zero? || name_only || @value.nil?

      if @value.is_a?(Array)
        @value.collect { |v| v.is_a?(Tensor) ? v.to_math(name_only, max_depth - 1) : v }
      else
        is_const ? @value : @name
      end
    end

    def auto_math(tensor, name_only = false, max_depth = 99, cur_depth = 0)
      tensor.is_a?(Tensor) ? tensor.to_math(name_only, max_depth, cur_depth) : tensor
    end

    def self.detect_type(value)
      if !!value == value
        :boolean
      elsif value.is_a?(String)
        :string
      elsif value.is_a?(Float)
        :float32
      elsif value.is_a?(Integer)
        :int32
      elsif value.is_a?(Array)
        detect_type(value[0])
      elsif value.is_a?(Tensor)
        value.data_type
      else
        :float32
      end
    end

    def self.cast_dtype(val, dtype)
      return val if dtype.nil?
      return val if val.is_a?(Tensor)

      if val.is_a?(Array)
        return val.collect do |v|
          cast_dtype(v, dtype)
        end
      end

      dtype = dtype[:dtype] if dtype.is_a?(Hash)

      case dtype.to_sym
      when :float64, :float32, :float16, :float
        if !!val == val
          val ? 1.0 : 0.0
        else
          val.to_f
        end
      when :string
        val.to_s
      when :uint32, :int32, :uint64, :uint16, :int16, :int64, :uint8, :int
        if !!val == val
          val ? 1 : 0
        else
          val.to_i
        end
      when :boolean
        !!val
      when :unknown
        val
      else
        raise "unknown data_type #{dtype} passed"
      end
    end

    def breakpoint!(&_block)
      self
    end

    def print!(message)
      _op(:print, self, self, message: message)
    end

    protected

    def setup_initial_state(options)
      @outputs = []
      @graph = options[:__graph] || TensorStream.get_default_graph
      @source = format_source(caller_locations)
    end

    def propagate_outputs
      # nop
    end

    def hashify_tensor(tensor)
      if tensor.is_a?(Tensor)
        tensor.to_h
      elsif tensor.is_a?(Array)
        tensor.collect { |t| hashify_tensor(t) }
      else
        tensor
      end
    end

    def _reshape(arr, shape)
      if arr.is_a?(Array)
        return arr if shape.size < 2

        slice = shape.shift
        arr.each_slice(slice).collect do |s|
          _reshape(s, shape)
        end
      else
        return arr if shape.empty?

        slice = shape.shift
        return arr if slice.nil?

        Array.new(slice) do
          _reshape(arr, shape.dup)
        end
      end
    end

    def build_name
      @is_const ? "Const#{graph.get_const_counter}:#{@rank}" : "Variable#{graph.get_var_counter}:#{@rank}"
    end
  end
end
