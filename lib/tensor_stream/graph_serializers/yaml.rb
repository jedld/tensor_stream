module TensorStream
  # Parses pbtext files and loads it as a graph
  class Yaml < TensorStream::Serializer
    include TensorStream::StringHelper
    include TensorStream::OpHelper

    def get_string(tensor_or_graph, session = nil, graph_keys = nil)
      graph = tensor_or_graph.is_a?(Tensor) ? tensor_or_graph.graph : tensor_or_graph
      serialized_arr = []

      node_keys = graph_keys.nil? ? graph.node_keys : graph.node_keys.select { |k| graph_keys.include?(k) }

      node_keys.each do |k|
        node = if block_given?
                 yield graph, k
               else
                 graph.get_tensor_by_name(k)
               end
        next unless node.is_a?(Operation)

        serialized_arr << node.to_h
      end

      serialized_arr.to_yaml
    end
  end
end