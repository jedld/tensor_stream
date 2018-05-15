module TensorStream
  # A class that defines a TensorStream graph
  class Graph
    attr_accessor :nodes, :collections, :eager_execution

    def initialize
      @eager_execution = false
      @nodes = {}
      @collections = {
        :"#{GraphKeys::GLOBAL_VARIABLES}" => []
      }
    end

    def reset
      @placeholder_counter = 0
      @const_counter = 0
      @var_counter = 0
      @op_counter = 0
      
      @nodes = {}
      @collections = {
        :"#{GraphKeys::GLOBAL_VARIABLES}" => []
      }
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

    def add_node(node)
      raise 'Placeholder cannot be used when eager_execution is enabled' if @eager_execution && node.is_a?(Placeholder)
      node.name = uniqunify(node.name) if @nodes[node.name]
      @nodes[node.name] = node
      node.send(:propagate_consumer, node)
      node.value = node.eval if @eager_execution
    end

    def node_added?(name)
      @nodes.key?(name)
    end

    def get_node(name)
      @nodes[name]
    end

    def add_node!(name, node)
      @nodes[name] = node
      node
    end

    def add_variable(node, options = {})
      raise "duplicate variable detected #{node.name} and reuse=false in current scope" if @nodes[node.name] && !options[:reuse]

      add_to_collection(GraphKeys::GLOBAL_VARIABLES, node)
      add_to_collection(GraphKeys::TRAINABLE_VARIABLES, node) if node.trainable?
      add_node(node)
    end

    def control_dependencies(_dependencies = [], &_block)
      raise 'not implemented'
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

    protected

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
