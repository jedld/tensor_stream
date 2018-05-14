module TensorStream
  module EvaluatorHelpers
    # explicit broadcasting helper
    def broadcast_dimensions(input, dims = [])
      return input if dims.empty?

      n = dims.shift

      if input.is_a?(Array)
        element = input.each_with_index.collect do |item, index|
          broadcast_dimensions(item, dims.dup)
        end

        if n
          element + Array.new(n) { element }.flatten(1)
        else
          element
        end
      else
        Array.new(n) do
          broadcast_dimensions(input, dims.dup)
        end
      end
    end

          # handle 2 tensor math operations
    def vector_op(vector, vector2, op = ->(a, b) { a + b }, switch = false)
      if get_rank(vector) < get_rank(vector2) # upgrade rank of A
        duplicated = Array.new(vector2.size) do
          vector
        end
        return vector_op(duplicated, vector2, op, switch)
      end

      return op.call(vector, vector2) unless vector.is_a?(Array)

      vector.each_with_index.collect do |item, index|
        next vector_op(item, vector2, op, switch) if item.is_a?(Array) && get_rank(vector) > get_rank(vector2)

        z = if vector2.is_a?(Array)
              if index < vector2.size
                vector2[index]
              else
                raise 'incompatible tensor shapes used during op' if vector2.size != 1
                vector2[0]
              end
            else
              vector2
            end

        if item.is_a?(Array)
          vector_op(item, z, op, switch)
        else
          switch ? op.call(z, item) : op.call(item, z)
        end
      end
    end
  
  end
end