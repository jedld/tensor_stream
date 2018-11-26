module TensorStream
  class Freezer
    def save(filename, tensor)
      serializer = TensorStream::Pbtext.new

      serialize_list = []
      visited = Set.new
      queue = Queue.new

      queue << tensor

      until queue.empty?
        current_node = queue.pop
        serialize_list << current_node.name
        visited << current_node
        if current_node.is_a?(Operation)
          current_node.inputs.reject { |i| visited.include?(i) }.each { |i| queue << i }
        end
      end

      serialize_list.reverse!

      frozen_graph = TensorStream::Graph.new
      source_graph = tensor.graph
      serialize_list.each do |k|
      end
    end

    def restore(filename)
    end
  end
end