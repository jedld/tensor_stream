require 'ostruct'

module TensorStream
  # Base class that defines a tensor like interface
  class Tensor
    include OpHelper
    attr_reader :graph
    attr_accessor :name, :data_type, :shape, :rank, :native_buffer, :is_const,
                  :value, :breakpoint, :internal, :source, :given_name,
                  :consumers, :outputs, :device

    def initialize(data_type, rank, shape, options = {})
      setup_initial_state(options)
      @data_type = data_type
      @rank = rank
      @breakpoint = false
      @shape = TensorShape.new(shape, rank)
      @value = nil

      @is_const = options[:const] || false
      @internal = options[:internal]
      @name = [@graph.get_name_scope, options[:name] || build_name].compact.reject(&:empty?).join('/')
      @given_name = @name

      if options[:value]
        if options[:value].is_a?(Array)
          # check if single dimenstion array is passed
          options[:value] = reshape(options[:value], shape.reverse.dup) if shape.size >= 2 && !options[:value].empty? && !options[:value][0].is_a?(Array)

          @value = options[:value].map { |v| v.is_a?(Tensor) ? Tensor.cast_dtype(v, @data_type) : v }
        elsif !shape.empty?
          @value = reshape(Tensor.cast_dtype(options[:value], @data_type), shape.dup)
        else
          @value = Tensor.cast_dtype(options[:value], @data_type)
        end
      end

      @graph.add_node(self)
    end

    def internal?
      !!@internal
    end

    def dtype
      @data_type
    end

    def self.reset_counters
      @const_counter = 0
      @var_counter = 0
      @placeholder_counter = 0
    end

    def +(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:add, self, other)
    end

    def [](index)
      _op(:index, self, index)
    end

    def *(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:mul, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def **(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:pow, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def /(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:div, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def -(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:sub, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def -@
      _op(:negate, self, nil)
    end

    def %(other)
      TensorStream.mod(self, other)
    end

    def floor
      TensorStream.floor(self)
    end

    def ceil
      TensorStream.ceil(self)
    end

    def zero?
      _op(:equal, self, TensorStream.constant(0, dtype: data_type, name: 'equal/is_zero?'))
    end

    def ==(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:equal, self, other)
    end

    def <(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:less, self, other)
    end

    def !=(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:not_equal, self, other)
    end

    def >(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:greater, self, other)
    end

    def >=(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:greater_equal, self, other)
    end

    def <=(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:less_equal, self, other)
    end

    def and(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:logical_and, self, other)
    end

    def matmul(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:mat_mul, self, other)
    end

    def dot(other)
      _a, other = TensorStream.check_data_types(self, other)
      _op(:mat_mul, self, other)
    end

    ##
    # Apply a reduction to tensor
    def reduce(op_type)
      reduce_op = case op_type.to_sym
                  when :+
                    :sum
                  when :*
                    :prod
                  else
                    raise "unsupported reduce op type #{op_type}"
                  end
      raise "blocks are not supported for tensors" if block_given?

      _op(reduce_op, self, nil)
    end

    def collect(&block)
      @value.collect(&block)
    end

    def to_s
      @name
    end

    def op
      @op ||= is_const ? _op(:const, self, nil, name: name) : _op(:variable, self, nil, name: name)
    end

    def eval(options = {})
      Session.default_session.run(self, options)
    end

    def to_h
      {
        name: @name,
        value: hashify_tensor(@value),
        dtype: @data_type,
        shape: @shape,
        const: !!is_const,
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

    def add_consumer(consumer)
      @consumers ||= Set.new
      @consumers << consumer.name if consumer.name != name
    end

    def setup_output(consumer)
      @outputs << consumer.name unless @outputs.include?(consumer.name)
    end

    def propagate_consumer(consumer)
      add_consumer(consumer)
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

    def reshape(arr, shape)
      if arr.is_a?(Array)
        return arr if shape.size < 2
        slice = shape.shift
        arr.each_slice(slice).collect do |s|
          reshape(s, shape)
        end
      else
        return arr if shape.empty?
        slice = shape.shift
        return arr if slice.nil?

        Array.new(slice) do
          reshape(arr, shape.dup)
        end
      end
    end

    def build_name
      @is_const ? "Const#{graph.get_const_counter}:#{@rank}" : "Variable#{graph.get_var_counter}:#{@rank}"
    end
  end
end
