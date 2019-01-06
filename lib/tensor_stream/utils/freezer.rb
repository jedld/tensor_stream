module TensorStream
  class Freezer
    include TensorStream::OpHelper

    ##
    # Utility class to convert variables to constants for production deployment
    #
    def convert(session, checkpoint_folder, output_file)
      model_file = File.join(checkpoint_folder, 'model.yaml')
      TensorStream.graph.as_default do |current_graph|
        YamlLoader.new.load_from_string(File.read(model_file))
        saver = TensorStream::Train::Saver.new
        saver.restore(session, checkpoint_folder)
        output_buffer = TensorStream::Yaml.new.get_string(current_graph) do |graph, node_key|
          node = graph.get_tensor_by_name(node_key)
          if node.operation == :variable_v2
            value = node.container
            options = {
              value: value,
              data_type: node.data_type,
              shape: shape_eval(value)
            }
            const_op = TensorStream::Operation.new(current_graph, inputs: [], options: options)
            const_op.name = node.name
            const_op.operation = :const
            const_op.data_type = node.data_type
            const_op.shape = TensorShape.new(shape_eval(value))

            const_op
          else
            node
          end
        end
        File.write(output_file, output_buffer)
      end
    end
  end
end