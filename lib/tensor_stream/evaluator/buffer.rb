module TensorStream
  # this class represents an evaluator specific native buffer
  class Buffer
    attr_accessor :data_type, :buffer, :dirty, :name

    def initialize(data_type:, buffer:)
      @data_type = data_type
      @buffer = buffer
    end

    def to_ruby
      buffer
    end
  end
end
