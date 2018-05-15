module TensorStream
  class Device
    attr_accessor :name
    def initialize(name)
      @name = name
    end
  end
end