module TensorStream
  class Dimension
    def initialize(dim)
      @dim = dim
    end

    def value
      @dim
    end
  end
  # class that defines a shape for TensorFlow compatibility
  class TensorShape
    attr_accessor :rank, :shape

    def initialize(new_shape, rank = nil)
      @shape = new_shape.is_a?(Array) ? new_shape : (new_shape ? [new_shape] : nil)
      @rank = rank.nil? && shape ? shape.size : rank
    end

    def to_s
      return "?" if @shape.nil?

      dimensions = @shape.collect { |r|
        "Dimension(#{r})"
      }.join(",")
      "TensorShape([#{dimensions}])"
    end

    def [](index)
      if index.is_a?(Range)
        TensorShape.new(@shape[index])
      else
        new_shape = @shape[index]
        Dimension.new(new_shape)
      end
    end

    def concatenate(other_shape)
      other_shape = TensorShape.as_shape(other_shape)
      return TensorShape.new(nil) if ndims.nil? || other_shape.ndims.nil?

      new_shape = @shape + other_shape.shape
      TensorShape.new(new_shape)
    end

    ##
    # Returns a shape based on `self` with at least the given rank.
    def with_rank_at_least(rank)
      raise TensorStream::ValueError, "Shape #{self} must have rank at least #{rank}" if @rank && @rank < rank

      self
    end

    def ndims
      shape ? shape.size : nil
    end

    def scalar?
      known? && shape.size.zero?
    end

    def known?
      return false if shape.nil?

      a_shape = shape.is_a?(Array) ? shape : [shape]
      a_shape.each { |s| return false if s.nil? || s < 0 }

      true
    end

    def fully_defined?
      known?
    end

    def merge_with(other)
      assert_compatible_with(other)

      if @shape.nil?
        TensorShape.new(other)
      else
        TensorShape.new(@shape)
      end
    end

    def compatible_with?(other)
      other = as_dimension(other)
      shape.nil? || other.nil? || shape == other
    end

    def as_dimension(value)
      value.is_a?(TensorShape) ? value.shape : value
    end

    def value
      shape
    end

    def dims
      shape.map { |s| Dimension.new(s) }
    end

    ##
    # Raises an exception if `other` is not compatible with this shape.
    def assert_compatible_with(other)
      raise TensorStream::ValueError, "Dimensions #{self} and #{other} are not compatible" unless compatible_with?(other)
    end

    ##
    #  Converts the given object to a TensorShape.
    def self.as_shape(shape)
      return shape if shape.is_a?(TensorShape)
      TensorShape.new(shape)
    end

    def self.unknown_shape(ndims: nil)
      if ndims.nil?
        TensorShape.new(nil)
      else
        TensorShape.new([nil] * ndims)
      end
    end

    def self.infer_shape(shape_a, shape_b)
      return nil if shape_a.nil? || shape_b.nil?
      return shape_a if shape_b.empty?
      return shape_b if shape_a.empty?
      return shape_a if shape_a == shape_b
      return shape_b if shape_b.size > shape_a.size
      return shape_a if shape_a.size > shape_b.size

      reversed_a = shape_a.reverse
      reversed_b = shape_b.reverse

      reversed_a.each_with_index.collect { |s, index|
        next s if index >= reversed_b.size
        next nil if s.nil? || reversed_b[index].nil?
        next nil if s.is_a?(Tensor) || reversed_b[index].is_a?(Tensor)
        next reversed_b[index] if reversed_b[index] > s

        s
      }.reverse
    end

    def self.reshape(arr, new_shape)
      arr = arr.is_a?(Array) ? arr.flatten : [arr]
      new_shape = new_shape.is_a?(TensorShape) ? new_shape.shape : new_shape
      new_shape = TensorShape.fix_inferred_elements(new_shape, arr.size)
      return arr[0] if arr.size == 1 && new_shape.empty?

      new_shape = new_shape.dup

      s = new_shape.shift

      if new_shape.size.zero?
        raise "reshape dimen mismatch #{arr.size} != #{s}" if arr.size != s

        return arr
      end

      dim = (arr.size / s)
      return arr if dim.zero?

      arr.each_slice(dim).collect do |slice|
        reshape(slice, new_shape.dup)
      end
    end

    def self.fix_inferred_elements(shape, total_size)
      return shape if shape.empty?
      return nil if shape[0].is_a?(Tensor)

      current_size = shape.inject(1) { |product, n| n > 0 ? product * n : product }
      inferred_size = total_size.nil? ? nil : total_size / current_size
      shape.map { |s| s == -1 ? inferred_size : s }
    end
  end
end
