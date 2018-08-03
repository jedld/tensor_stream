# A tensorstream device
module TensorStream
  # Class that describes a supported device
  class Device
    attr_accessor :name, :type, :evaluator
    def initialize(name, type, evaluator)
      @name = name
      @type = type
      @evaluator = evaluator
    end
  end
end
