module TensorStream
  class Initializer
    attr_writer :op
    def initialize(op)
      @op = op
    end

    def op
      @op.call
    end
  end
end