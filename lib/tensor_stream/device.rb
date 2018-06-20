# A tensorstream device
module TensorStream
  class Device
    attr_accessor :name, :type, :evaluator
    def initialize(name, type, evaluator)
      @name = name
      @type = type
      @evaluator = evaluator
    end
  end
end