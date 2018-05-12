module TensorStream
  # Defines a TensorStream controlflow op
  class ControlFlow < Operation
    attr_accessor :ops

    def initialize(flow_type, items, ops = nil, options = {})
      @graph = options[:graph] || TensorStream.get_default_graph

      @operation = :"flow_#{flow_type}"
      @items = items
      @name = set_name
      @ops = ops
      @source = format_source(caller_locations)

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
