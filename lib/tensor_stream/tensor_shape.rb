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

    def known?
      return false if shape.nil?
      shape.each { |s| return false if s.nil? }

      true
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
        next nil if s.is_a?(Tensor) || reversed_b[index].is_a?(Tensor)
        next reversed_b[index] if reversed_b[index] > s
        s
      end.reverse
    end

    def self.reshape(arr, new_shape)
      return arr if new_shape.empty?

      s = new_shape.shift

      if new_shape.size.zero?
        raise "reshape dimen mismatch #{arr.size} != #{s}" if arr.size != s
        return arr
      end

      dim = (arr.size / s)
      arr.each_slice(dim).collect do |slice|
        reshape(slice, new_shape.dup)
      end
    end

    def self.fix_inferred_elements(shape, total_size)
      return shape if shape.empty?

      current_size = shape.inject(1) { |product, n| n > 0 ? product * n : product }
      inferred_size = total_size / current_size
      shape.map { |s| s == -1 ? inferred_size : s }
    end
  end
end
