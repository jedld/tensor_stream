module TensorStream
  # Defines a TensorStream controlflow op
  class ControlFlow < Operation
    attr_accessor :ops

    def initialize(flow_type, inputs, ops = nil, options = {})
      setup_initial_state(options)
      @options = options
      @operation = :"flow_#{flow_type}"
      @inputs = inputs
      @name = [@graph.get_name_scope, options[:name] || set_name].compact.join('/')
      @ops = ops
      @consumers = Set.new
      @shape = TensorShape.new([inputs.size])
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
