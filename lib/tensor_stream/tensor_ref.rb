module TensorStream
  class TensorRef < Tensor
    attr_reader :skip_cache

    def initialize(initial_value)
      @value = initial_value.op
      @data_type = @value.data_type
      @skip_cache = true
    end

    def update_ref(ref)
      @value = ref.op
      @data_type = @value.data_type
    end

    def op
      self
    end

    def device
      @value.device
    end

    def shape
      @value.shape
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