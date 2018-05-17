module TensorStream
  # class that defines a shape for TensorFlow compatibility
  class TensorShape
    attr_accessor :rank, :shape

    def initialize(shape, rank = nil)
      @shape = shape
      @rank = rank.nil? && shape ? shape.size : rank
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

    def self.infer_shape(shape_a, shape_b)
      return shape_a if shape_b.nil?
      return shape_b if shape_a.nil?
      return shape_a if shape_a == shape_b
      return shape_b if shape_b.size > shape_a.size
      return shape_a if shape_a.size > shape_b.size

      reversed_a = shape_a.reverse
      reversed_b = shape_b.reverse

      reversed_a.each_with_index.collect do |s, index|
        next s if index >= reversed_b.size
        next nil if s.nil? || reversed_b[index].nil?
        next reversed_b[index] if reversed_b[index] > s
        s
      end.reverse
    end
  end
end
