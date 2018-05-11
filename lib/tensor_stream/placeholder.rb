module TensorStream
  class Placeholder < Tensor
    def initialize(data_type, rank, shape, options = {})
      @data_type = data_type
      @rank = rank
      @shape = TensorShape.new(shape, rank)
      @value = nil
      @is_const = false
      @source = set_source(caller_locations)
      @graph = options[:graph] || TensorStream.get_default_graph
      @name = options[:name] || build_name
      @graph.add_node(self)
    end

    private

    def build_name
      "Placeholder#{Tensor.placeholder_name}:#{@rank}"
    end
  end
end