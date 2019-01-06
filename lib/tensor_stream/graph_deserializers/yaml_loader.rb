module TensorStream
  ##
  # Class for deserialization from a YAML file
  class YamlLoader
    def initialize(graph = nil)
      @graph = graph || TensorStream.get_default_graph
    end

    ##
    # Loads a model Yaml file and builds the model from it
    #
    # Args:
    # filename: String - Location of Yaml file
    #
    # Returns: Graph where model is restored to
    def load_from_file(filename)
      load_from_string(File.read(filename))
    end

    ##
    # Loads a model Yaml file and builds the model from it
    #
    # Args:
    # buffer: String - String in Yaml format of the model
    #
    # Returns: Graph where model is restored to
    def load_from_string(buffer)
      serialized_ops = YAML.safe_load(buffer, [Symbol], aliases: true)
      serialized_ops.each do |op_def|
        inputs = op_def[:inputs].map { |i| @graph.get_tensor_by_name(i) }
        options = {}

        new_var = nil
        if op_def.dig(:attrs, :container)
          new_var = Variable.new(op_def.dig(:attrs, :data_type))
          var_shape = op_def.dig(:attrs, :container, :shape)
          var_options = op_def.dig(:attrs, :container, :options)
          var_options[:name] = op_def[:name]

          new_var.prepare(var_shape.size, var_shape, TensorStream.get_variable_scope, var_options)
          options[:container] = new_var

          @graph.add_variable(new_var, var_options)
        end

        new_op = Operation.new(@graph, inputs: inputs, options: op_def[:attrs].merge(options))
        new_op.operation = op_def[:op].to_sym
        new_op.name = op_def[:name]
        new_op.shape = TensorShape.new(TensorStream::InferShape.infer_shape(new_op))
        new_op.rank = new_op.shape.rank
        new_op.data_type = new_op.set_data_type(op_def.dig(:attrs, :data_type))
        new_op.is_const = new_op.infer_const
        new_op.given_name = new_op.name
        new_var.op = new_op if new_var

        @graph.add_node(new_op)
      end
      @graph
    end
  end
end