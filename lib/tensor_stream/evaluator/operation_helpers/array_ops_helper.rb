module TensorStream
  # varoius utility functions for array processing
  module ArrayOpsHelper
    def slice_tensor(input, start, size)
      return input if size.empty?
      start_index = start.shift
      dimen_size = start_index + size.shift

      input[start_index...dimen_size].collect do |item|
        if item.is_a?(Array)
          slice_tensor(item, start.dup, size.dup)
        else
          item
        end
      end
    end

    def truncate(input, target_shape)
      rank = get_rank(input)
      return input if rank.zero?

      start = Array.new(rank) { 0 }
      slice_tensor(input, start, target_shape)
    end

    def reduced_shape(input_shape, axes)
      return [] if axes.nil? # reduce to scalar
      axes = [ axes ] unless axes.is_a?(Array)
      return input_shape if axes.empty?

      axes.each do |dimen|
        input_shape[dimen] = 1
      end
      input_shape
    end

    def broadcast(input_a, input_b)
      sa = shape_eval(input_a)
      sb = shape_eval(input_b)

      return [input_a, input_b] if sa == sb

      # descalar
      if sa.empty?
        input_a = [input_a]
        sa = [1]
      end

      if sb.empty?
        input_b = [input_b]
        sb = [1]
      end

      target_shape = shape_diff(sa, sb)

      if target_shape
        input_b = broadcast_dimensions(input_b, target_shape)
      else
        target_shape = shape_diff(sb, sa)
        raise "Incompatible shapes for op #{shape_eval(input_a)} vs #{shape_eval(input_b)}" if target_shape.nil?

        input_a = broadcast_dimensions(input_a, target_shape)
      end

      [input_a, input_b]
    end

    # explicit broadcasting helper
    def broadcast_dimensions(input, dims = [])
      return input if dims.empty?

      d = dims.shift

      if input.is_a?(Array) && (get_rank(input) - 1) == dims.size
        row_to_dup = input.collect do |item|
          broadcast_dimensions(item, dims.dup)
        end

        row_to_dup + Array.new(d) { row_to_dup }.flatten(1)
      elsif input.is_a?(Array)
        Array.new(d) { broadcast_dimensions(input, dims.dup) }
      else
        Array.new(d + 1) { input }
      end
    end

    # handle 2 tensor math operations
    def vector_op(vector, vector2, op = ->(a, b) { a + b }, switch = false, safe = true)
      if get_rank(vector) < get_rank(vector2) # upgrade rank of A
        duplicated = Array.new(vector2.size) do
          vector
        end
        return vector_op(duplicated, vector2, op, switch)
      end

      return op.call(vector, vector2) unless vector.is_a?(Array)

      vector.each_with_index.collect do |item, index|
        next vector_op(item, vector2, op, switch) if item.is_a?(Array) && get_rank(vector) > get_rank(vector2)

        if safe && vector2.is_a?(Array)
          next nil if vector2.size != 1 && index >= vector2.size
        end

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
      end.compact
    end

    def shape_diff(shape_a, shape_b)
      return nil if shape_b.size > shape_a.size

      reversed_a = shape_a.reverse
      reversed_b = shape_b.reverse

      reversed_a.each_with_index.collect do |s, index|
        next s if index >= reversed_b.size
        return nil if reversed_b[index] > s
        s - reversed_b[index]
      end.reverse
    end

    def tile_arr(input, dimen, multiples)
      t = multiples[dimen]
      if dimen == multiples.size - 1
        return nil if t.zero?
        input * t # ruby array dup
      else
        new_arr = input.collect do |sub|
          tile_arr(sub, dimen + 1, multiples)
        end.compact

        return nil if new_arr.empty?

        new_arr * t
      end
    end

    def process_function_op(a, op)
      # ruby scalar
      if (a.is_a?(Tensor) && a.shape.rank > 0) || a.is_a?(Array)
        vector_op(a, 0, op)
      else
        op.call(a, 0)
      end
    end

    def get_rank(value, rank = 0)
      return rank unless value.is_a?(Array)
      return rank + 1 if value.empty?

      get_rank(value[0], rank + 1)
    end

    def softmax(arr)
      return arr if arr.empty?

      sum = if !arr[0].is_a?(Array)
        arr.map { |a| Math.exp(a - arr.max) }.reduce(:+)
      end

      arr.collect do |item|
        if item.is_a?(Array)
          softmax(item)
        else
          Math.exp(item - arr.max) / sum
        end
      end
    end

    def softmax_grad(arr)
      return arr if arr.empty?

      arr.each_with_index.collect do |item, index|
        if item.is_a?(Array)
          softmax_grad(item)
        else
          arr.each.each_with_index.collect do |item2, index2|
            if index != index2
              -item * item2
            else
              item * (1.0 - item)
            end
          end
        end
      end
    end
  end
end