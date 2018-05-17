require 'ostruct'

module TensorStream
  # Base class that defines a tensor like interface
  class Tensor
    include OpHelper

    attr_accessor :name, :data_type, :shape, :rank, :native_buffer, :is_const,
                  :value, :breakpoint, :internal, :source, :given_name, :graph,
                  :consumers

    def initialize(data_type, rank, shape, options = {})
      @data_type = data_type
      @rank = rank
      @breakpoint = false
      @shape = TensorShape.new(shape, rank)
      @value = nil
      @source = format_source(caller_locations)
      @is_const = options[:const] || false
      @internal = options[:internal]
      @graph = options[:graph] || TensorStream.get_default_graph
      @name = [@graph.get_name_scope, options[:name] || build_name].compact.reject(&:empty?).join('/')
      @given_name = @name

      if options[:value]
        if options[:value].is_a?(Array)
          # check if single dimenstion array is passed
          options[:value] = reshape(options[:value], shape.reverse.dup) if shape.size >= 2 && !options[:value].empty? && !options[:value][0].is_a?(Array)

          @value = options[:value].collect do |v|
            v.is_a?(Tensor) ? Tensor.cast_dtype(v, data_type) : v
          end
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
      TensorStream::Operation.new(:add, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def [](index)
      TensorStream::Operation.new(:index, self, index)
    end

    def *(other)
      TensorStream::Operation.new(:mul, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def **(other)
      TensorStream::Operation.new(:pow, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def /(other)
      TensorStream::Operation.new(:div, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def -(other)
      TensorStream::Operation.new(:sub, self, TensorStream.convert_to_tensor(other, dtype: data_type))
    end

    def -@
      TensorStream::Operation.new(:negate, self, nil)
    end

    def ==(other)
      _op(:equal, self, other)
    end

    def <(other)
      _op(:less, self, other)
    end

    def !=(other)
      _op(:not_equal, self, other)
    end

    def >(other)
      _op(:greater, self, other)
    end

    def >=(other)
      _op(:greater_equal, self, other)
    end

    def <=(other)
      _op(:less_equal, self, other)
    end

    def and(other)
      _op(:logical_and, self, other)
    end

    def matmul(other)
      _op(:matmul, self, other)
    end
    

    def dot(other)
      _op(:matmul, self, other)
    end

    def collect(&block)
      @value.collect(&block)
    end

    def to_s
      @name
    end

    def op
      is_const ? _op(:const, self, nil, name: self.name) : _op(:variable, self, nil, name: self.name)
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

    def auto_math(tensor, name_only = false, max_depth = 99, _cur_depth = 0)
      tensor.is_a?(Tensor) ? tensor.to_math(name_only, max_depth, _cur_depth) : tensor
    end

    def self.detect_type(value)
      if value.is_a?(String)
        :string
      elsif value.is_a?(Float)
        :float32
      elsif value.is_a?(Integer)
        :int32
      elsif value.is_a?(Array)
        :array
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

      case dtype.to_sym
      when :float32, :float
        if !!val == val
          val ? 1.0 : 0.0
        else
          val.to_f
        end
      when :string
        val.to_s
      when :int32, :int16
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

    def breakpoint!(&block)
      self
    end

    def print!(message)
      _op(:print, self, self, message: message)
    end

    protected

    def add_consumer(consumer)
      @consumers ||= []
      @consumers << consumer.name if !@consumers.include?(consumer.name) && consumer.name!=self.name
    end

    def propagate_consumer(consumer)
      add_consumer(consumer)
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
