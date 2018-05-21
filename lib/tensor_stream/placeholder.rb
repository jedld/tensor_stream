module TensorStream
  # Class that defines a TensorStream placeholder
  class Placeholder < Tensor
    def initialize(data_type, rank, shape, options = {})
      setup_initial_state(options)

      @data_type = data_type
      @rank = rank
      @shape = TensorShape.new(shape, rank)
      @value = nil
      @is_const = false

      @name = [@graph.get_name_scope, options[:name] || build_name].compact.reject(&:empty?).join('/')
      @graph.add_node(self)
    end

    private

    def build_name
      "Placeholder#{graph.get_placeholder_counter}:#{@rank}"
    end
  end
end
