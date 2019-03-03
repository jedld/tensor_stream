class Array
  include TensorStream::MonkeyPatch

  def /(other)
    TensorStream.convert_to_tensor(self) * other
  end

  def %(other)
    TensorStream.convert_to_tensor(self) % other
  end

  def **(other)
    TensorStream.convert_to_tensor(self)**other
  end

  def max_index
    if first.is_a?(Float)
      highest = first
      highest_index = 0
      each_with_index do |item, index|
        next if item.nan?

        if item > highest
          highest = item
          highest_index = index
        end
      end
      highest_index
    else
      index(max)
    end
  end

  def min_index
    if first.is_a?(Float)
      highest = first
      highest_index = 0
      each_with_index do |item, index|
        next if item.nan?

        if item < highest
          highest = item
          highest_index = index
        end
      end
      highest_index
    else
      index(min)
    end
  end
end
