module TensorStream
  # Class that defines a TensorStream placeholder
  class Placeholder < Tensor
    def initialize(data_type, rank, shape, options = {})
      @graph = options[:graph] || TensorStream.get_default_graph

      @data_type = data_type
      @rank = rank
      @shape = TensorShape.new(shape, rank)
      @value = nil
      @is_const = false
      @source = format_source(caller_locations)

      @name = options[:name] || build_name
      @graph.add_node(self)
    end

    private

    def build_name
      "Placeholder#{graph.get_placeholder_counter}:#{@rank}"
    end
  end
end
