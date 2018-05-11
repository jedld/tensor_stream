module TensorStream
  class TensorShape
    attr_accessor :rank, :shape
    
    def initialize(shape, rank)
      @shape = shape
      @rank = rank
    end

    def to_s
      dimensions = @shape.collect do |r|
        "Dimension(#{r})"
      end.join(',')
      "TensorShape([#{dimensions}])"
    end

    def [](index)
      @shape[index]
    end
    
    def ndims
      shape.size
    end
  end
end