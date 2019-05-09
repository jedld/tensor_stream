require "tensor_stream/helpers/infer_shape"
module TensorStream
  # TensorStream class that defines an operation
  class Operation < Tensor
    include OpHelper

    attr_accessor :name, :operation, :inputs, :rank, :device, :control_inputs, :control_flow_context, :consumers, :breakpoint
    attr_reader :outputs, :options, :is_const, :data_type, :shape

    def initialize(graph, inputs:, options:)
      @consumers = Set.new
      @outputs = []
      @op = self
      @graph = graph
      @inputs = inputs
      @options = options
      @control_flow_context = nil
      @control_inputs = []
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

    def add_control_inputs(ops)
      if ops
        ops.each do |op|
          raise TensorStream::TypeError, "op must be an Operation: #{op}"
          assert_same_graph(op)
          @control_inputs << op
        end
        recompute_node_def
      end
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

    def to_math(name_only = false, max_depth = 20, cur_depth = 0)
      return @name if max_depth.zero?
      return { } if cur_depth > max_depth
      input_representation = inputs.compact.map { |t| t.to_math(name_only, max_depth, cur_depth + 1) }
      { op: operation, name: @name, in: input_representation, value: options[:value] }
    end

    def run
      eval
    end

    def op
      self
    end

    protected

    def recompute_node_def
      #TODO: does something related protobuf node definition
    end

    def assert_same_graph(original_item, item)
      raise TensorStream::ValueError, "#{item} must be from the same graph as #{original_item}"
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
