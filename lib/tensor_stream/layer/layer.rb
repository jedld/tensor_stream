module TensorStream
  class Layer
    attr_accessor :trainable, :name
    include TensorStream::StringHelper

    def initialize(options = {}, trainable: true, name: nil, dtype: nil, dynamic: false)
      allowed_kwargs = %i[input_shape batch_input_shape batch_size weights activity_regularizer]

      options.keys.each do |kwarg|
        raise TensorStream::TypeError, "Keyword argument not understood:#{kwarg}" unless allowed_kwargs.include?(kwarg)
      end

      @trainable = trainable
      @stateful = false
      # Indicates whether `build` needs to be called upon layer call, to create
      # the layer's weights.
      @built = false
      # Provides information about which inputs are compatible with the layer.
      @input_spec = nil
      @supports_masking = false
      init_set_name(name)
      @activity_regularizer = kwargs.dig(:activity_regularizer, nil)
      @trainable_weights = []
      @non_trainable_weights = []
      @updates = []
      @callable_losses = []
      # A list of Tensors containing activity regularizers and losses manually
      # added through `add_loss`.
      @losses = []
      @dtype = dtype.nil ? nil : dtypes.to_sym
      @inbound_nodes = []
      @outbound_nodes = []
    end

    protected

    def init_set_name(name, zero_based: true)
      @name = if name.nil?
                unique_layer_name(underscore(self.class.to_s), zero_based: zero_based)
              else
                name
              end
    end

    ##
    # Makes a layer name (or arbitrary string) unique within a TensorFlow graph.
    # Arguments:
    #   name: String name to make unique.
    #   name_uid_map: An optional Hash to use when creating unique
    #                 names. If None (default), uses a per-Graph dictionary.
    #   avoid_names: An optional set or hash with names which should not be used. If
    #                 None (default) does not avoid any names.
    #   namespace: Gets a name which is unique within the (graph, namespace). Layers
    #              which are not Networks use a blank namespace and so get graph-global
    #              names.
    #   zero_based: If True, name sequences start with no suffix (e.g. "dense",
    #               "dense_1"). If False, naming is one-based ("dense_1", "dense_2").
    # Returns:
    # Unique string name.
    def unique_layer_name(name, name_uid_map: nil, avoid_names: nil, namespace: '', zero_based: false)
      name_uid_map = default_graph_uid_map if name_uid_map.nil?
      avoid_names = Set.new if avoid_names.nil?
      proposed_name = nil
      while proposed_name.nil? || avoid_names.include?(proposed_name)
        name_key = [namespace, name]
        if zero_based
          number = name_uid_map[name_key]
          proposed_name = if number
                            name + '_' + number.to_s
                          else
                            name
                          end
          name_uid_map[name_key] += 1
        else
          name_uid_map[name_key] += 1
          proposed_name = name + '_' + name_uid_map[name_key].to_s
        end
      end

      proposed_name
    end

    def default_graph_uid_map
      graph = TensorStream.get_default_graph
      name_uid_map = Graph.layer_name_uids.fetch(graph.object_id, nil)
      if name_uid_map.nil?
        name_uid_map = Hash.new(0)
        Graph.layer_name_uids[graph.object_id] = name_uid_map
      end
      name_uid_map
    end
  end
end