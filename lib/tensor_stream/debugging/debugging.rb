module TensorStream
  module Debugging
    extend TensorStream::OpHelper

    def add_check_numerics_ops
      graph = TensorStream.get_default_graph
      nodes_to_process = graph.nodes.values.select { |node| node.is_a?(Operation) }

      nodes_to_process.each do |node|
        node.inputs = node.inputs.collect do |input|
          next if input.nil?
          next input if input.is_a?(Variable)

          if input.is_a?(Tensor) && TensorStream::Ops::FLOATING_POINT_TYPES.include?(input.data_type)
            TensorStream.check_numerics(input, "#{node.name}/#{input.name}", name: "check/#{node.name}/#{input.name}" )
          else
            input
          end
        end
      end
    end
  end
end