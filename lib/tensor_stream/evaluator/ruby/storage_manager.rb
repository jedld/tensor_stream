module TensorStream
  class RubyStorageManager
    def self.current_storage_manager
      @storage_manager ||= RubyStorageManager.new
    end

    def initialize
      @variables = {}
    end

    def exists?(graph, name)
      return false if !@variables.key?(graph.object_id)

      @variables[graph.object_id].key?(name.to_sym)
    end

    def create_variable(graph, name, value)
      raise "no name specified" if name.nil?

      @variables[graph.object_id][name.to_sym] = value
    end

    def assign_value(graph, name, value)
      raise "no name specified" if name.nil?

      @variables[graph.object_id] ||= {}
      @variables[graph.object_id][name.to_sym] = value
    end

    def read_value(graph, name)
      raise "no name specified" if name.nil?

      @variables[graph.object_id][name.to_sym]
    end

    def clear_variables(graph)
      @variables[graph.object_id] = {}
    end
  end
end