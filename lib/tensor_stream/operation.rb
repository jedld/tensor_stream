require "tensor_stream/helpers/infer_shape"
module TensorStream
  # TensorStream class that defines an operation
  class Operation < Tensor
    include OpHelper

    attr_accessor :name, :operation, :inputs, :rank, :device, :consumers, :breakpoint
    attr_reader :outputs, :options, :is_const, :data_type, :shape

    def initialize(graph, inputs: [], options: {})
      @consumers = Set.new
      @outputs = []
      @op = self
      @graph = graph
      @inputs = inputs
      @options = options
    end

    def inspect
      "Op(#{operation} name: #{name} shape: #{@shape || "?"} data_type: #{data_type})"
    end

    def to_s
      @name
    end

    def to_h
      {
        op: operation.to_s,
        name: name.to_s,
        data_type: @data_type,
        inputs: @inputs.map(&:name),
        attrs: serialize_options,
      }
    end

    def const_value
      @options ? @options[:value] : nil
    end

    def container_buffer
      @options[:container] ? @options[:container].buffer : nil
    end

    def container
      @options[:container].read_value
    end

    def container=(value)
      @options[:container].value = value
    end

    def set_input(index, value)
      @inputs[index] = value
      @shape = TensorShape.new(TensorStream::InferShape.infer_shape(self))
      @rank = @shape.rank
      @is_const = infer_const
      @data_type = set_data_type(@options[:data_type])
    end

    def set_option(key, value)
      @options.merge!(key.to_sym => value)
    end

    def infer_const
      return false if breakpoint

      case operation
      when :random_standard_normal, :random_uniform, :truncated_normal, :glorot_uniform, :print, :check_numerics
        false
      when :const
        true
      when :placeholder
        false
      when :variable_v2
        false
      else
        non_const = @inputs.compact.find { |input| !input.is_const }
        non_const ? false : true
      end
    end

    def set_name
      @operation.to_s
    end

    def set_data_type(passed_data_type)
      case operation
      when :where
        @inputs[1].data_type
      when :case
        if @inputs[2]
          @inputs[2].data_type
        else
          @inputs[1].data_type
        end
      when :case_grad
        @inputs[2].data_type
      when :placeholder, :variable_v2, :const
        options[:data_type]
      when :fill
        @inputs[1].data_type
      when :logical_and
        :boolean
      when :shape, :rank, :shape_n
        options[:out_type] || :int32
      when :zeros, :ones
        options[:dtype] || :float32
      when :random_standard_normal, :random_uniform, :glorot_uniform, :truncated_normal
        passed_data_type || :float32
      when :concat
        @inputs[1].data_type
      when :conv2d_backprop_input
        @inputs[1].data_type
      when :index
        if @inputs[0].is_a?(ControlFlow)
          if @inputs[1].is_const
            @inputs[0].inputs[@inputs[1].const_value].data_type
          else
            :unknown
          end
        else
          @inputs[0].data_type
        end
      else
        OpMaker.infer_data_type(self, self, passed_data_type)
      end
    end

    def to_math(name_only = false, max_depth = 99, cur_depth = 0)
      return @name if max_depth.zero?

      sub_input = auto_math(inputs[0], name_only, max_depth - 1, cur_depth + 1)
      sub_input2 = auto_math(inputs[1], name_only, max_depth - 1, cur_depth + 1) if inputs[1]

      out = case operation
            when :argmax
              "argmax(#{sub_input},#{options[:axis]})"
            when :negate
              "-#{sub_input}"
            when :index
              "#{sub_input}[#{sub_input2}]"
            when :slice
              "#{sub_input}[#{sub_input2}]"
            when :assign_sub
              "(#{inputs[0] ? inputs[0].name : "self"} -= #{auto_math(inputs[1], name_only, 1)})"
            when :assign_add
              "(#{inputs[0] ? inputs[0].name : "self"} += #{auto_math(inputs[1], name_only, 1)})"
            when :assign
              "(#{inputs[0] ? inputs[0].name : "self"} = #{auto_math(inputs[1], name_only, 1)})"
            when :sin, :cos, :tanh
              "#{operation}(#{sub_input})"
            when :add
              "(#{sub_input} + #{sub_input2})"
            when :sub
              "(#{sub_input} - #{sub_input2})"
            when :pow
              "(#{sub_input}^#{sub_input2})"
            when :div
              "(#{sub_input} / #{sub_input2})"
            when :mul
              if auto_math(inputs[0]) == 1
                sub_input2
              elsif auto_math(inputs[1]) == 1
                sub_input
              else
                "(#{sub_input} * #{sub_input2})"
              end
            when :sum
              "sum(|#{sub_input}|,  axis=#{sub_input2})"
            when :mean
              "mean(|#{sub_input}|, axis=#{sub_input2})"
            when :prod
              "prod(|#{sub_input}|,  axis=#{sub_input2})"
            when :gradients
              "gradient(#{sub_input})"
            when :stop_gradient
              sub_input
            when :mat_mul
              "#{sub_input}.matmul(#{sub_input2})"
            when :eye
              "eye(#{sub_input})"
            when :transpose
              "transpose(#{sub_input})"
            when :shape
              "#{sub_input}.shape"
            when :exp
              "e^#{sub_input})"
            when :ones
              "ones(#{sub_input})"
            when :ones_like
              "ones_like(#{sub_input})"
            when :flow_group
              "flow_group(#{inputs.collect { |i| auto_math(i, name_only, max_depth - 1, cur_depth) }.join(",")})"
            when :zeros
              "zeros(#{sub_input})"
            when :reshape
              "reshape(#{sub_input},#{sub_input2})"
            when :rank
              "#{sub_input}.rank"
            when :less
              "#{sub_input} < #{sub_input2}"
            when :less_equal
              "#{sub_input} <= #{sub_input2}"
            when :greater
              "#{sub_input} > #{sub_input2}"
            when :greater_equal
              "#{sub_input} >= #{sub_input2}"
            when :square
              "#{sub_input}\u00B2"
            when :log
              "log(#{sub_input})"
            when :identity
              "identity(#{sub_input})"
            when :print
              "print(#{sub_input})"
            when :pad
              "pad(#{sub_input},#{auto_math(options[:paddings])})"
            when :equal
              "#{sub_input} == #{sub_input2}"
            when :not_equal
              "#{sub_input} != #{sub_input2}"
            when :logical_and
              "#{sub_input} && #{sub_input2}"
            when :sqrt
              "sqrt(#{sub_input})"
            when :log1p
              "log1p(#{sub_input})"
            when :zeros_like
              "zeros_like(#{sub_input})"
            when :where
              "where(#{auto_math(options[:pred], name_only, max_depth - 1, cur_depth)}, #{sub_input}, #{sub_input2})"
            when :max
              "max(#{sub_input},#{sub_input2})"
            when :cast
              "cast(#{sub_input}, #{data_type})"
            when :broadcast_transform
              "broadcast_transform(#{sub_input},#{sub_input2})"
            when :broadcast_gradient_args
              "broadcast_transform(#{sub_input},#{sub_input2})"
            else
              "#{operation}(#{sub_input})" if sub_input
              "#{operation}(#{sub_input}, #{sub_input2})" if sub_input && sub_input2
      end
      ["\n", Array.new(cur_depth + 1) { " " }, out].flatten.join
    end

    def run
      eval
    end

    def op
      self
    end

    private

    def serialize_options
      excludes = %i[internal_name source]

      @options.reject { |k, v| excludes.include?(k) || v.nil? }.map { |k, v|
        v = case v.class.to_s
            when "TensorStream::TensorShape"
              v.shape
            when "Array"
              v
            when "String", "Integer", "Float", "Symbol", "FalseClass", "TrueClass"
              v
            when "TensorStream::Variable"
              {name: v.name, options: v.options, shape: v.shape.shape.dup}
            else
              raise "unknown type #{v.class}"
        end
        [k.to_sym, v]
      }.to_h
    end

    def add_consumer(consumer)
      @consumers << consumer.name if consumer.name != name
    end

    def setup_output(consumer)
      @outputs << consumer.name unless @outputs.include?(consumer.name)
    end

    def propagate_consumer(consumer)
      add_consumer(consumer)
      @inputs.compact.each do |input|
        if input.is_a?(Array)
          input.flatten.compact.map(&:op).select { |t| t.is_a?(Tensor) }.each do |t|
            next if t.consumers.include?(consumer.name)

            t.send(:propagate_consumer, consumer)
          end
        elsif input.name != name && !input.consumers.include?(consumer.name)
          input.send(:propagate_consumer, consumer)
        end
      end
    end

    def propagate_outputs
      @inputs.compact.each do |input|
        if input.is_a?(Array)
          input.flatten.compact.map(&:op).each do |t|
            t.send(:setup_output, self) if t.is_a?(Tensor)
          end
        elsif input.is_a?(Tensor) && (input.name != name)
          input.send(:setup_output, self)
        end
      end
    end

    def setup_initial_state(options)
      @outputs = []
      @graph = options[:__graph] || TensorStream.get_default_graph
      @source = format_source(caller_locations)
    end
  end
end
