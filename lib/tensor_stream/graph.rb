module TensorStream
  # A class that defines a TensorStream graph
  class Graph
    include OpHelper

    attr_accessor :nodes, :collections, :eager_execution, :random_seed,
                  :constants, :unfeedable, :control_flow_context
    attr_reader :node_keys

    def initialize
      @eager_execution = false
      @nodes = {}
      @unfeedable = Set.new
      @node_keys = []
      @collections = {
        :"#{GraphKeys::GLOBAL_VARIABLES}" => [],
        :"#{GraphKeys::TRAINABLE_VARIABLES}" => [],
      }
      @constants = {}
      @control_flow_context = nil
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
        :"#{GraphKeys::TRAINABLE_VARIABLES}" => [],
      }
      @constants = {}
    end

    def as_default
      Thread.current[:tensor_stream_current_graph_queue] ||= []
      Thread.current[:tensor_stream_current_graph_queue] << Graph.get_default_graph

      Thread.current[:tensor_stream_current_graph] = self
      yield(self) if block_given?
      Thread.current[:tensor_stream_current_graph] = Thread.current[:tensor_stream_current_graph_queue].pop
      self
    end

    def self.layer_name_uids
      @layer_uids ||= {}
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

    def prevent_feeding(tensor)
      @unfeedable << tensor
    end

    def add_node(node, name = nil)
      raise "Placeholder cannot be used when eager_execution is enabled" if @eager_execution && node.is_a?(Placeholder)

      if name.nil?
        node.name = if @nodes[node.name]
          uniquenify(node.name)
        else
          node.name
        end
      end

      node.device = get_device_scope
      @node_keys << node.name
      @nodes[node.name] = node
      @constants[node.name] = node if node.is_const

      node.send(:propagate_outputs)
      node.send(:propagate_consumer, node)
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

    def [](name)
      get_node(name)
    end

    def add_node!(name, node)
      @nodes[name] = node
      node
    end

    def add_op(operation, *args)
      options = if args.last.is_a?(Hash)
        args.pop
      else
        {}
      end

      inputs = args.map { |i| TensorStream.convert_to_tensor(i) }.map { |i| i ? i.op : nil }

      new_op = Operation.new(self, inputs: inputs, options: options)
      new_op.source = format_source(caller_locations)
      new_op.operation = operation
      new_op.set_shape(TensorStream::InferShape.infer_shape(new_op))
      new_op.rank = new_op.shape.rank
      new_op.name = options[:internal_name] || [get_name_scope, options[:name] || set_operation_name(new_op)].compact.reject(&:empty?).join("/")
      new_op.internal = options[:internal]

      new_op.data_type = new_op.set_data_type(options[:data_type])
      new_op.is_const = new_op.infer_const

      new_op.given_name = new_op.name

      new_op
    end

    def add_op!(operation, *args)
      add_op(operation, *args).tap { |node| add_node(node) }
    end

    def set_operation_name(op)
      op.operation.to_s
    end

    def unique_name(name)
      uniquenify(name)
    end

    def mutation_lock
      # no op for now
      yield
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

      node
    end

    def add_variable!(node, options = {})
      node = add_variable(node, options)
      op = Graph.get_default_graph.add_op!(:variable_v2, container: node, internal_name: node.name, shape: options[:shape], data_type: options[:data_type])
      node.name = op.name
      op
    end

    def control_dependencies(control_inputs = [])
      Thread.current["ts_graph_#{object_id}"] ||= {}
      Thread.current["ts_graph_#{object_id}"][:control_dependencies] ||= []
      Thread.current["ts_graph_#{object_id}"][:control_dependencies] << Graph.get_default_graph.add_op!(:no_op, *control_inputs)
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

      name = @op_counter.zero? ? "" : "_#{@op_counter}"

      @op_counter += 1

      name
    end

    def get_placeholder_counter
      @placeholder_counter ||= 0
      @placeholder_counter += 1

      return "" if @placeholder_counter == 1

      "_#{@placeholder_counter}"
    end

    def get_var_counter
      @var_counter ||= 0
      @var_counter += 1

      return "" if @var_counter == 1
      "_#{@var_counter}"
    end

    def get_const_counter
      @const_counter ||= 0

      name = @const_counter.zero? ? "" : "_#{@const_counter}"

      @const_counter += 1
      name
    end

    def get_name_scope
      graph_thread_storage = Thread.current["ts_graph_#{object_id}"]
      return nil if graph_thread_storage.nil? || graph_thread_storage[:current_scope].nil?

      graph_thread_storage[:current_scope].join("/")
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
      return VariableScope.new(name: "", reuse: false, initializer: nil) if Thread.current[:tensor_stream_variable_scope].nil? || Thread.current[:tensor_stream_variable_scope].empty?
      scope = Thread.current[:tensor_stream_variable_scope].last
      scope
    end

    def uniquenify(name)
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
