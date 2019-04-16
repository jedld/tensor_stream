module TensorStream
  module BaseLayerUtils
    def default_graph_uid_map
      graph = TensorStream.get_default_graph
      name_uid_map = Graph.layer_name_uids.fetch(graph.object_id, nil)
      if name_uid_map.nil?
        name_uid_map = Hash.new(0)
        Graph.layer_name_uids[graph.object_id] = name_uid_map
      end
      name_uid_map
    end

    def make_variable(name, shape: nil, dtype: :float32, initializer: nil, trainable: nil, validate_shape: true)
      initializing_from_value = false
      init_val = nil
      variable_dtype = nil
      initializing_from_value = true if !initializer && !initializer.is_a?(Proc)
      TensorStream.init_scope do
        if initializing_from_value
          TensorStream.variable(initializer, name: name, trainable: trainable)
        else
          TensorStream.variable(nil, initializer: initializer, name: name, trainable: trainable, dtype: dtype)
        end
      end
    end
  end
end