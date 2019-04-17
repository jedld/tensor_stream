require 'tensor_stream/layer/trackable'
require 'tensor_stream/layer/base_module'
require 'tensor_stream/layer/base_layer_utils'

module TensorStream
  class Layer < BaseModule
    attr_accessor :trainable, :name
    include TensorStream::StringHelper
    include TensorStream::BaseLayerUtils

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
      @activity_regularizer = options.dig(:activity_regularizer, nil)
      @trainable_weights = []
      @non_trainable_weights = []
      @updates = []
      @callable_losses = []
      # A list of Tensors containing activity regularizers and losses manually
      # added through `add_loss`.
      @losses = []
      @dtype = dtype.nil? ? nil : dtypes.to_sym
      @inbound_nodes = []
      @outbound_nodes = []

      if options.key?(:input_shape) || options.key?(:batch_input_shape)
        if options.key?(:batch_input_shape)
          @batch_input_shape = options[:batch_input_shape].freeze
        elsif options.key?(:input_shape)
          if options.key?(:batch_size)
            @batch_size =  options[:batch_size]
          else
            @batch_size = nil
          end
          @batch_input_shape = [@batch_size] + options[:input_shape].freeze
        end
      end

      if options.key?(:weights)
        @initial_weights = options[:weights]
      else
        @initial_weights = nil
      end
    end

    def build(input_shape)
      @built = true
    end

    def call(inputs, *args)
      inputs
    end

    ##
    # Adds a new variable to the layer
    def add_weight(options = {}, name: nil, shape: nil, dtype: :float, initializer: nil, regularizer: nil,
      trainable: true,
      constraint: nil,
      partitioner: nil,
      use_resource: nil)
      @shape = [] if shape.nil?
      options.each do |k,v|
        raise TensorStream::TypeError, "Unkonwn keyword argument #{k}" if ![:getter, :collections, :experimental_autocast].include?(k)
      end
      getter = options.fetch(:getter, nil)
      collections = options.fetch(:collections, nil)
      autocast = options.fetch(:experimental_autocast, nil)

      if initalizer.nil?
        initializer = if fp_type?(@data_type)
                        TensorStream.glorot_uniform_initializer
                      elsif int_type?(@data_type)
                        TensorStream.zeros_initializer
                      else
                      raise TensorStream::ValueError, "An initializer for variable #{name} of type #{@data_type} for layer #{@name}"
                      end
      end

      variable = add_variable_with_custom_getter(name: name, shape: shape, getter: getter || make_variable, overwrite: true,
        initializer: initializer, dtype: dtype, constraint: constraint, trainable: trainable && @trainable)
      track_variable(variable)
      if trainable
        @trainable_weights << variable
      else
        @non_trainable_weights << variable
      end

      variable
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
  end
end