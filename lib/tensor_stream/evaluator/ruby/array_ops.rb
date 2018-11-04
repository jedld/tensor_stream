module TensorStream
  module ArrayOps
    def ArrayOps.included(klass)
      klass.class_eval do
        register_op :slice do |context, tensor, inputs|
          input = inputs[0]
          start = inputs[1]
          size = complete_eval(tensor.options[:size], context)
          raise "start index and size not of the same shape #{start.size} != #{size.size}" if start.size != size.size

          slice_tensor(input, start.dup, size.dup)
        end

        register_op %i[flow_dynamic_stitch dynamic_stitch] do |_context, _tensor, inputs|
          indexes, data = inputs
          merged = []
          merge_dynamic_stitch(merged, indexes, data)
          merged
        end

        register_op :gather do |_context, _tensor, inputs|
          params, indexes = inputs
          gather(params, indexes)
        end

        register_op %i[concat concat_v2] do |_context, _tensor, inputs|
          axis = inputs.shift
          concat_array(inputs, axis)
        end

        register_op :stack do |_context, tensor, inputs|
          axis = tensor.options[:axis] || 0
          shape = shape_eval(inputs[0])
          rank = shape.size + 1
          elem_size = shape.empty? ? 1 : shape.reduce(:*)
          output_buffer = Array.new(inputs.size * elem_size) { 0 }
          new_shape = [inputs.size]
          shape.inject(new_shape) { |ns, s| ns << s }

          divisors = new_shape.dup.drop(1).reverse.inject([1]) do |a, s|
            a << s * a.last
          end.reverse

          axis = rank + axis if axis < 0
          rotated_shape = Array.new(axis + 1) { new_shape.shift }
          new_shape = rotated_shape.rotate! + new_shape

          multipliers = new_shape.dup.drop(1).reverse.inject([1]) do |a, s|
            a << s * a.last
          end.reverse

          inputs.each_with_index do |input, index|
            raw_input = input.is_a?(Array) ? input.flatten : [input]
            start = index * divisors.first

            raw_input.each_with_index do |x, index2|
              index_map = []
              ptr = start + index2
              divisors.each_with_object(index_map) do |div, a|
                a << (ptr / div.to_f).floor
                ptr = ptr % div
              end

              rotated_index = Array.new(axis + 1) { index_map.shift }
              index_map = rotated_index.rotate! + index_map

              ptr2 = 0
              multipliers.each_with_index do |m, idx|
                ptr2 += index_map[idx] * m
              end

              output_buffer[ptr2] = x
            end
          end

          TensorShape.reshape(output_buffer, new_shape)
        end

        register_op :unstack do |_context, tensor, inputs|
          value = inputs[0]

          axis = tensor.options[:axis] || 0
          new_shape = shape_eval(inputs[0])
          rank = new_shape.size - 1

          divisors = new_shape.dup.drop(1).reverse.inject([1]) do |a, s|
            a << s * a.last
          end.reverse

          axis = rank + axis if axis < 0
          rotated_shape = Array.new(axis + 1) { new_shape.shift }
          new_shape = rotated_shape.rotate!(-1) + new_shape
          output_buffer = Array.new(new_shape.reduce(:*)) { 0 }

          multipliers = new_shape.dup.drop(1).reverse.inject([1]) do |a, s|
            a << s * a.last
          end.reverse

          inputs.each_with_index do |input, index|
            raw_input = input.is_a?(Array) ? input.flatten : [input]
            start = index * divisors.first

            raw_input.each_with_index do |x, index2|
              index_map = []
              ptr = start + index2
              divisors.each_with_object(index_map) do |div, a|
                a << (ptr / div.to_f).floor
                ptr = ptr % div
              end

              rotated_index = Array.new(axis + 1) { index_map.shift }
              index_map = rotated_index.rotate!(-1) + index_map

              ptr2 = 0
              multipliers.each_with_index do |m, idx|
                ptr2 += index_map[idx] * m
              end

              output_buffer[ptr2] = x
            end
          end

          res = TensorShape.reshape(output_buffer, new_shape)

          TensorStream::Evaluator::OutputGroup.new(res, res.map { tensor.inputs[0].data_type })
        end

        register_op :squeeze do |_context, tensor, inputs|
          val = inputs[0]
          shape = shape_eval(val)

          axis = !tensor.options[:axis].is_a?(Array) ? [tensor.options[:axis]] : tensor.options[:axis]

          if !axis.empty?
            axis.each do |x|
              if shape[x] == 1
                shape[x] = nil
              else
                raise TensorStream::ValueError, "unable to squeeze dimension that does not have a size of 1"
              end
            end
          else
            shape = shape.map { |s| s == 1 ? nil : s }
          end

          TensorShape.reshape(val, shape.compact)
        end

        register_op :expand_dims do |_context, _tensor, inputs|
          val, axis = inputs
          axis = axis.nil? ? 0 : axis

          shape = shape_eval(val)
          axis = -axis if axis == shape.size

          new_shape = shape.dup.insert(axis, 1).compact

          TensorShape.reshape([val], new_shape)
        end

        register_op :fill do |_context, tensor, inputs|
          shape = inputs[0] || tensor.shape.shape
          value = inputs[1]

          func = -> { value }

          if shape.is_a?(Array) && shape.size.zero?
            func.call
          else
            shape = [shape.to_i] unless shape.is_a?(Array)
            generate_vector(shape, generator: func)
          end
        end

        register_op :invert_permutation do |_context, _tensor, inputs|
          input = inputs[0]
          output = input.dup

          unless input.nil?
            input.size.times.each do |index|
              output[input[index]] = index
            end
          end

          output
        end

        register_op :index, no_eval: true do |_context, _tensor, inputs|
          f = inputs[0]
          index = inputs[1]
          if f.is_a?(TensorStream::Evaluator::OutputGroup)
            f.outputs[index]
          else
            f[index]
          end
        end

        register_op :setdiff1d do |_context, tensor, inputs|
          input, remove = inputs
          idx = []
          out = []
          input.each_with_index do |x, index|
            next if remove.include?(x)

            out << x
            idx << index
          end
          idx = idx.map { |i| Tensor.cast_dtype(i, tensor.options[:index_dtype]) } unless tensor.options[:index_dtype] == :int32
          TensorStream::Evaluator::OutputGroup.new([out, idx], tensor.inputs.map(&:data_type))
        end

        register_op :size do |_context, tensor, inputs|
          input = inputs[0]
          Tensor.cast_dtype(input.flatten.size, tensor.options[:out_type])
        end

        register_op :range do |_context, _tensor, inputs|
          start, limit, delta = inputs
          raise " delta !=0 " if delta.zero?
          raise " Requires start <= limit when delta > 0" if (start > limit) && delta > 0
          raise " Requires start >= limit when delta < 0" if (start < limit) && delta < 0

          cur_step = start
          r = []
          Kernel.loop do
            break if start == limit
            break if (start < limit) && (cur_step >= limit)
            break if (start > limit) && (cur_step <= limit)

            r << cur_step
            cur_step += delta
          end
          r
        end

        register_op :eye do |_context, tensor, inputs|
          rows, columns = inputs

          Array.new(rows) do |i|
            Array.new(columns) do |col|
              if fp_type?(tensor.data_type)
                i == col ? 1.0 : 0.0
              else
                i == col ? 1 : 0
              end
            end
          end
        end

        register_op %i[zeros ones zeros_like ones_like] do |_context, tensor, inputs|
          shape = if %i[zeros_like ones_like].include?(tensor.operation)
                    shape_eval(inputs[0])
                  else
                    inputs[0] || tensor.shape.shape
                  end

          func = if %i[zeros zeros_like].include?(tensor.operation)
                   -> { int_type?(tensor.data_type) ? 0 : 0.0 }
                 else
                   -> { int_type?(tensor.data_type) ? 1 : 1.0 }
                 end
          if shape.is_a?(Array) && shape.size.zero?
            func.call
          else
            shape = [shape.to_i] unless shape.is_a?(Array)

            cache_key = "#{tensor.operation}_#{shape}"
            if @context[:_cache].key?(cache_key)
              @context[:_cache][cache_key]
            else
              generate_vector(shape, generator: func).tap do |v|
                @context[:_cache][cache_key] = v
              end
            end
          end
        end

        register_op :truncate do |_context, _tensor, inputs|
          truncate(inputs[0], inputs[1])
        end

        register_op :rank do |_context, _tensor, inputs|
          get_rank(inputs[0])
        end

        register_op :split  do |_context, tensor, inputs|
          value, num_split, axis = inputs

          value_shape = shape_eval(value)
          res = if num_split.is_a?(Array)
                  begin_index = 0
                  num_split.collect do |num|
                    end_index = begin_index + num
                    arr = split_tensor(value, begin_index, end_index, axis)
                    begin_index = end_index
                    arr
                  end
                else
                  raise TensorStream::ValueError, "#{num_split} does not divide #{value_shape[axis]} evenly" if value_shape[axis] % num_split != 0

                  piece_sizes = value_shape[axis] / num_split
                  Array.new(num_split) do |num|
                    begin_index = num * piece_sizes
                    end_index = begin_index + piece_sizes
                    split_tensor(value, begin_index, end_index, axis)
                  end
                end
          TensorStream::Evaluator::OutputGroup.new(res, res.map { tensor.inputs[0].data_type })
        end

        register_op :reshape do |_context, _tensor, inputs|
          arr, new_shape = inputs
          arr = [arr] unless arr.is_a?(Array)
          flat_arr = arr.flatten
          if new_shape.size.zero? && flat_arr.size == 1
            flat_arr[0]
          else
            TensorShape.reshape(flat_arr, new_shape)
          end
        end

        register_op :pad do |context, tensor, inputs|
          arr_pad(inputs[0], inputs[1], tensor.data_type)
        end

        register_op :tile do |_context, _tensor, inputs|
          input, multiples = inputs
          rank = get_rank(input)
          raise '1D or higher tensor required' if rank.zero?
          raise "invalid multiple size passed #{rank} != #{multiples.size}" if rank != multiples.size

          tile = tile_arr(input, 0, multiples)
          tile.nil? ? [] : tile
        end

        register_op :cond, noop: true do |context, tensor, inputs|
          pred = global_eval(tensor, tensor.options[:pred], context)

          if all_true?(pred)
            global_eval(tensor, inputs[0], context)
          else
            global_eval(tensor, inputs[1], context)
          end
        end

        register_op %i[select where] do |context, tensor, inputs|
          pred = complete_eval(tensor.options[:pred], context)
          call_3way_vector_op(pred, inputs[0], inputs[1], context, ->(t, u, v) { t ? u : v })
        end

        register_op :shape do |_context, tensor, inputs|
          shape_eval(inputs[0], tensor.options[:out_type])
        end

        register_op :shape_n do |_context, tensor, inputs|
          shapes = inputs.collect do |input|
            shape_eval(input)
          end
          TensorStream::Evaluator::OutputGroup.new(shapes, shapes.map { tensor.options[:out_type] })
        end
      end
    end
  end
end