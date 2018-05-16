module TensorStream
  class Initializer
    attr_accessor :op
    def initialize(op)
      @op = op
    end
  end
end