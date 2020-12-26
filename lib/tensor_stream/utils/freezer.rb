module TensorStream
  class Freezer
    include TensorStream::OpHelper

    ##
    # Utility class to convert variables to constants for production deployment
    #
    def convert(session, checkpoint_folder, output_file)
      model_file = File.join(checkpoint_folder, "model.yaml")
      TensorStream.graph.as_default do |current_graph|
        YamlLoader.new.load_from_string(File.read(model_file))
        saver = TensorStream::Train::Saver.new
        saver.restore(session, checkpoint_folder)

        # collect all assign ops and remove them from the graph
        remove_nodes = Set.new(current_graph.nodes.values.select { |op| op.is_a?(TensorStream::Operation) && op.operation == :assign }.map { |op| op.consumers.to_a }.flatten.uniq)

        output_buffer = TensorStream::Yaml.new.get_string(current_graph) { |graph, node_key|
          node = graph.get_tensor_by_name(node_key)
          case node.operation
          when :variable_v2
            value = Evaluator.read_variable(node.graph, node.options[:var_name])
           if value.nil?
             raise "#{node.options[:var_name]} has no value"
           end

            options = {
              value: value,
              data_type: node.data_type,
              shape: shape_eval(value),
            }
            const_op = TensorStream::Operation.new(current_graph, inputs: [], options: options)
            const_op.name = node.name
            const_op.operation = :const
            const_op.data_type = node.data_type
            const_op.shape = TensorShape.new(shape_eval(value))

            const_op
          when :assign
            nil
          else
            remove_nodes.include?(node.name) ? nil : node
          end
        }
        File.write(output_file, output_buffer)
      end
    end
  end
end
