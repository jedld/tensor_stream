module TensorStream
  # Defines a TensorStream controlflow op
  class DynamicStitch < Operation
    attr_accessor :ops

    def initialize(flow_type, inputs, ops = nil, options = {})
      setup_initial_state(options)

      @operation = :"flow_#{flow_type}"
      @options = options.merge(n: inputs[0].size)
      @inputs = inputs.flatten(1).map { |i| TensorStream.convert_to_tensor(i) }.map { |i| i ? i.op : nil }

      @consumers = Set.new
      @data_type = Tensor.detect_type(inputs[1])
      @name = [@graph.get_name_scope, options[:name] || set_name].compact.join('/')
      @ops = ops
      @shape = TensorShape.new(nil)
      @graph.add_node(self)
    end

    def set_data_type(_passed_data_type)
      :unknown
    end

    def run
      eval
    end
  end
end