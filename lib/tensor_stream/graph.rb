module TensorStream
  # A class that defines a TensorStream graph
  class Graph
    attr_accessor :nodes, :node_keys, :collections, :eager_execution, :random_seed, :constants

    def initialize
      @eager_execution = false
      @nodes = {}
      @node_keys = []
      @collections = {
        :"#{GraphKeys::GLOBAL_VARIABLES}" => [],
        :"#{GraphKeys::TRAINABLE_VARIABLES}" => []
      }
      @constants = {}
    end

    def reset
      @placeholder_counter = 0
      @const_counter = 0
      @var_counter = 0
      @op_counter = 0
      @random_seed = nil
      @nodes = {}
      @node_keys = []
      @collections = {
        :"#{GraphKeys::GLOBAL_VARIABLES}" => [],
        :"#{GraphKeys::TRAINABLE_VARIABLES}" => []
      }
      @constants = {}
    end

    def as_default
      Thread.current[:tensor_stream_current_graph] = self
      yield(self) if block_given?
      self
    end

    def name_scope(name = nil)
      Thread.current["ts_graph_#{object_id}"] ||= {}
      Thread.current["ts_graph_#{object_id}"][:current_scope] ||= []
      Thread.current["ts_graph_#{object_id}"][:current_scope] << name

      begin
        yield get_name_scope if block_given?
      ensure
        Thread.current["ts_graph_#{object_id}"][:current_scope].pop
      end
    end

    ##
    # Returns a context manager that specifies the default device to use.
    def device(device_name)
      Thread.current["ts_graph_#{object_id}"] ||= {}
      Thread.current["ts_graph_#{object_id}"][:default_device] ||= []
      Thread.current["ts_graph_#{object_id}"][:default_device] << device_name
      begin
        yield
      ensure
        Thread.current["ts_graph_#{object_id}"][:default_device].pop
      end
    end

    def self.get_default_graph
      Thread.current[:tensor_stream_current_graph] || create_default
    end

    def self.create_default
      Thread.current[:tensor_stream_current_graph] = TensorStream::Graph.new
    end

    def get_collection(name, _options = {})
      @collections[name.to_sym]
    end

    def add_to_collection(collection_name, val)
      @collections[collection_name.to_sym] ||= []
      @collections[collection_name.to_sym] << val
    end

    def add_node(node, name = nil)
      raise 'Placeholder cannot be used when eager_execution is enabled' if @eager_execution && node.is_a?(Placeholder)

      if name.nil?
        node.name = if @nodes[node.name]
                      uniqunify(node.name)
                    else
                      node.name
                    end
      end

      node.device = get_device_scope
      @node_keys << node.name
      @nodes[node.name] = node
      @constants[node.name] = node if node.is_const
      # puts "adding node"
      node.send(:propagate_outputs)
      node.send(:propagate_consumer, node)
      # puts "#{node.name}"
      node.value = node.eval if @eager_execution
    end

    def node_added?(name)
      @nodes.key?(name)
    end

    def get_node(name)
      @nodes[name]
    end

    def get_tensor_by_name(name)
      raise TensorStream::KeyError, "#{name} not found" unless @nodes.key?(name)
      get_node(name)
    end

    def add_node!(name, node)
      @nodes[name] = node
      node
    end

    def add_variable(node, options = {})
      scope = _variable_scope

      raise "duplicate variable detected #{node.name} and reuse=false in current scope" if @nodes[node.name] && !scope.reuse
      return @nodes[node.name] if @nodes[node.name]
      raise "shape is not declared for #{node.name}" if node.shape.nil?

      if !options[:collections].nil? && !options[:collections].empty?
        options[:collections] = [options[:collections]] unless options[:collections].is_a?(Array)
        options[:collections].each { |coll| add_to_collection(coll, node) }
      end

      add_to_collection(GraphKeys::GLOBAL_VARIABLES, node)
      add_to_collection(GraphKeys::TRAINABLE_VARIABLES, node) if node.trainable?

      op = Operation.new(:variable_v2, container: node, internal_name: node.name, shape: options[:shape], data_type: options[:data_type])
      node.name = op.name
      op
    end

    def control_dependencies(control_inputs = [])
      Thread.current["ts_graph_#{object_id}"] ||= {}
      Thread.current["ts_graph_#{object_id}"][:control_dependencies] ||= []
      Thread.current["ts_graph_#{object_id}"][:control_dependencies] << Operation.new(:no_op, *control_inputs)
      begin
        yield
      ensure
        Thread.current["ts_graph_#{object_id}"][:control_dependencies].pop
      end
    end

    def enable_eager_execution
      @eager_execution = true
    end

    def disable_eager_execution
      @eager_execution = false
    end

    def executing_eagerly?
      @eager_execution
    end

    def get_operation_counter
      @op_counter ||= 0

      name = @op_counter.zero? ? '' : "_#{@op_counter}"

      @op_counter += 1

      name
    end

    def get_placeholder_counter
      @placeholder_counter ||= 0
      @placeholder_counter += 1

      return '' if @placeholder_counter == 1
      "_#{@placeholder_counter}"
    end

    def get_var_counter
      @var_counter ||= 0
      @var_counter += 1

      return '' if @var_counter == 1
      "_#{@var_counter}"
    end

    def get_const_counter
      @const_counter ||= 0

      name = @const_counter.zero? ? '' : "_#{@const_counter}"

      @const_counter += 1
      name
    end

    def get_name_scope
      graph_thread_storage = Thread.current["ts_graph_#{object_id}"]
      return nil if graph_thread_storage.nil? || graph_thread_storage[:current_scope].nil?

      graph_thread_storage[:current_scope].join('/')
    end

    def get_dependency_scope
      graph_thread_storage = Thread.current["ts_graph_#{object_id}"]
      return nil if graph_thread_storage.nil? || graph_thread_storage[:control_dependencies].nil?
      graph_thread_storage[:control_dependencies].last
    end

    def get_device_scope
      graph_thread_storage = Thread.current["ts_graph_#{object_id}"]
      return :default if graph_thread_storage.nil? || graph_thread_storage[:default_device].nil?
      graph_thread_storage[:default_device].last
    end

    def as_graph_def
      TensorStream::Pbtext.new.get_string(self)
    end

    def self.parse_from_string(buffer)
      builder = TensorStream::GraphBuilder.new(Graph.new)
      builder.build(buffer)
    end

    def graph_def_versions
      "producer: 26"
    end

    protected

    def _variable_scope
      return VariableScope.new(name: '', reuse: false, initializer: nil) if Thread.current[:tensor_stream_variable_scope].nil? || Thread.current[:tensor_stream_variable_scope].empty?
      scope = Thread.current[:tensor_stream_variable_scope].last
      scope
    end

    def uniqunify(name)
      counter = 0
      new_name = name
      Kernel.loop do
        counter += 1
        new_name = "#{name}_#{counter}"

        break unless @nodes.key?(new_name)
      end
      new_name
    end
  end
end
