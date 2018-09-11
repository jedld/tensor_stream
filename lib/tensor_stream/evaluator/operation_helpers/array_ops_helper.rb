module TensorStream
  # varoius utility functions for array processing
  module ArrayOpsHelper
    def split_tensor(input, begin_index, end_index, axis = 0)
      if axis.zero?
        input[begin_index...end_index]
      else
        input.collect do |item|
          split_tensor(item, begin_index, end_index, axis - 1)
        end
      end
    end

    def slice_tensor(input, start, size)
      return input if size.empty?
      start_index = start.shift
      current_size = size.shift
      dimen_size = if current_size == -1
        input.size - 1
      else
        start_index + current_size - 1
      end

      input[start_index..dimen_size].collect do |item|
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

    def _reduced_shape(input_shape, axes)
      return [] if axes.nil? # reduce to scalar
      axes = [axes] unless axes.is_a?(Array)
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

      vector.each_with_index.collect do |input, index|
        next vector_op(input, vector2, op, switch) if input.is_a?(Array) && get_rank(vector) > get_rank(vector2)

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

        if input.is_a?(Array)
          vector_op(input, z, op, switch)
        else
          switch ? op.call(z, input) : op.call(input, z)
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

    def last_axis(arr)
      return arr if get_rank(arr) <= 2

      arr.inject([]).map do |sub, rows|
        rows + last_axis(sub)
      end
    end

    def softmax(arr)
      return arr if arr.empty?

      if !arr[0].is_a?(Array)
        c = arr.max
        arr = arr.map { |a| Math.exp(a - c) }
        sum = arr.reduce(:+)
        arr.collect do |input|
          input / sum
        end
      else
        arr.collect { |input| softmax(input) }
      end
    end

    def softmax_grad(arr)
      return arr if arr.empty?
      arr.each_with_index.collect do |input, index|
        if input.is_a?(Array)
          softmax_grad(input)
        else
          arr.each_with_index.collect do |input2, index2|
            if index != index2
              -input * input2
            else
              input * (1.0 - input)
            end
          end
        end
      end
    end

    def gather(params, indexes)
      indexes.collect do |index|
        if index.is_a?(Array)
          gather(params, index)
        else
          params[index]
        end
      end
    end

    # general case transposition with flat arrays
    def transpose_with_perm(arr, new_arr, shape, new_shape, perm)
      arr_size = shape.reduce(:*)
      divisors = shape.dup.drop(1).reverse.inject([1]) do |a, s|
        a << s * a.last
      end.reverse

      multipliers = new_shape.dup.drop(1).reverse.inject([1]) do |a, s|
        a << s * a.last
      end.reverse

      arr_size.times do |p|
        ptr = p
        index = []
        divisors.each_with_object(index) do |div, a|
          a << (ptr / div.to_f).floor
          ptr = ptr % div
        end

        # remap based on perm
        remaped = perm.map { |x| index[x] }

        ptr2 = 0
        multipliers.each_with_index do |m, idx|
          ptr2 += remaped[idx] * m
        end

        new_arr[ptr2] = arr[p]
      end

      [new_arr, new_shape]
    end

    def reduce_axis(current_axis, axis, val, keep_dims, f)
      return val unless val.is_a?(Array)

      r = val.collect do |v|
        reduce_axis(current_axis + 1, axis, v, keep_dims, f)
      end

      should_reduce_axis = axis.nil? || (axis.is_a?(Array) && axis.include?(current_axis)) || (current_axis == axis)

      if should_reduce_axis
        reduced_val = r[0]
        if r.size > 1
          reduced_val = f.call(r[0..val.size])
        elsif r.empty?
          reduced_val = f.call(nil)
        end
        keep_dims ? [reduced_val] : reduced_val
      else
        r
      end
    end

    def reduce(val, axis, keep_dims, func = nil)
      rank = get_rank(val)
      return val if axis && axis.is_a?(Array) && axis.empty?

      func = lambda do |arr|
        reduced_val = arr[0]
        arr[1..arr.size].each do |v|
          reduced_val = vector_op(reduced_val, v, ->(t, u) { t + u })
        end
        reduced_val
      end if func.nil?

      axis = if axis.nil?
               nil
             elsif axis.is_a?(Array)
               return val if axis.empty?

               axis.map { |a| a < 0 ? rank - a.abs : a }
             else
               axis < 0 ? rank - axis.abs : axis
             end

      reduce_axis(0, axis, val, keep_dims, func)
    end
  end
end