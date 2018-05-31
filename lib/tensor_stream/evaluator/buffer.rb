module TensorStream
  # this class represents an evaluator specific native buffer
  class Buffer
    attr_accessor :dirty, :name

    def to_ruby
      raise "not implemented"
    end
  end
end