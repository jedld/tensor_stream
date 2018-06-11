module TensorStream
  module Debugging
    extend TensorStream::OpHelper

    def add_check_numerics_ops
      graph = TensorStream.get_default_graph
      nodes_to_process  = graph.nodes.values.select { |node| node.is_a?(Operation) }

      nodes_to_process.each do |node|
        node.items = node.items.compact.collect do |item|
          if TensorStream::Ops::FLOATING_POINT_TYPES.include?(item.data_type)
            TensorStream.check_numerics(item, "#{node.name}/#{item.name}", name: "check/#{node.name}/#{item.name}" )
          else
            item
          end
        end
      end
    end
  end
end