module TensorStream
  class TensorRef < Tensor
    attr_reader :buffer, :op

    def initialize(initial_value, options = {})
      setup_initial_state(options)
      scope_name = "Ref"
      @name = [scope_name, build_name].compact.reject(&:empty?).join("/")
      @data_type = initial_value.data_type
      @value = initial_value
      @op = _op(:read_ref, initial_value, ref_name: @name)
    end

    def update_ref(ref)
      @value = ref.op
      @data_type = @value.data_type
    end

    def rank
      @value.rank
    end

    def op
      @op
    end

    def skip_cache
      false
    end

    def device
      @value.device
    end

    def shape
      @value.shape
    end

    def inspect
      "Variable(#{@name} shape: #{@shape || "?"} data_type: #{@data_type})"
    end

    protected

    def build_name
      "TensorRef#{graph.get_var_counter}:#{@rank}"
    end

    def setup_output(_out)
    end

    def consumers
      []
    end

    def propagate_consumer(consumer)
    end
  end
end