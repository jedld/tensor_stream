module TensorStream
  module NNUtils
    ##
    # transposes the batch and time dimensions of a Tensor.
    def _transpose_batch_time(x)
      x_static_shape = x.shape
      return x if !x_static_shape.rank.nil? && x_static_shape.rank < 2

      x_rank = TensorStream.rank(x)
      x_t = TensorStream.transpose(
          x, TensorStream.concat([[1, 0], TensorStream.range(2, x_rank)], axis: 0))

      x_t.set_shape(TensorStream::TensorShape.new([x_static_shape[1].value, x_static_shape[0].value]).
          concatenate(x_static_shape[2..x_static_shape.shape.size]))
      x_t
    end

    def _flatten(inputs)
      return inputs.flatten if inputs.is_a?(Array)
      return inputs.keys.sort.map { |k| inputs[k] } if inputs.is_a?(Hash)
      [inputs]
    end

    ##
    # Get static input batch size if available, with fallback to the dynamic one.
    def _best_effort_input_batch_size(flat_input)
      flat_input.each do |input|
        shape = input.shape
        next if shape.rank.nil?
        raise TensorStream::ValueError, "Expected input tensor #{input} to have rank at least 2" if shape.rank < 2

        batch_size = shape[1].value
        return batch_size unless batch_size.nil?
      end

      TensorStream.shape(flat_input[0])[1]
    end

    ##
    # Returns a given flattened sequence packed into a given structure.
    def pack_sequence_as(structure, flat_sequence)
      raise TensorStream::TypeError, "flat_sequence must be a sequence" unless flat_sequence.is_a?(Array)
      if !structure.is_a?(Array)
        raise TensorStream::ValueError, "Structure is a scalar but lflat_sequence.size == #{flat_sequence.size} > 1" if flat_sequence.size != 1
        return flat_sequence[0]
      end

      final_index, packed = _packed_nest_with_indices(structure, flat_sequence, 0)
      _sequence_like(structure, packed)
    end

    def map_structure(func, *structure, check_types: true)
      raise TensorStream::TypeError, "func must be callable, got: #{func}" unless func.is_a?(Proc)
      raise TensorStream::ValueError, "Must provide at least one structure" unless structure

      structure[1..structure.size - 1].each do |other|
        assert_same_structure(structure[0], other, check_types: check_types)
      end
      flat_structure = structure.map { |s| _flatten(s) }

      pack_sequence_as(structure[0], flat_structure.map { |x| func.call(x) })
    end

    ##
    # Concat that enables int, Tensor, or TensorShape values.
    def _concat(prefix, suffix, static: false)
      p_static = nil
      s_static = nil

      p = if prefix.is_a?(Tensor)
            p = prefix
            p_static = constant_value(prefix)
            if p.shape.ndims == 0
              TensorStream.expand_dims(p, 0)
            elsif p.shape.ndims != 1
               raise TensorStream::ValueError,"prefix tensor must be either a scalar or vector, " +
                    "but saw tensor: #{p}"
            end
          else
            p = TensorShape.as_shape(prefix)
            p_static = !p.ndims.nil? ? p.shape : nil
            p.fully_defined? ? TensorStream.constant(p.shape, dtype: :int32) : nil
          end

      s = if suffix.is_a?(Tensor)
            s = suffix
            s_static = constant_value(suffix)
            if s.shape.ndims == 0
              TensorStream.expand_dims(s, 0)
            elsif s.shape.ndims != 1
              raise TensorStream::ValueError, "suffix tensor must be either a scalar or vector, but saw tensor: #{s}"
            end
          else
            s = TensorShape.as_shape(suffix)
            s_static = !s.ndims.nil? ? s.shape : nil
            s.fully_defined? ? TensorStream.constant(s.shape, dtype: :int32) : nil
          end

      shape = if static
                shape = TensorShape.as_shape(p_static).concatenate(s_static)
                !shape.ndims.nil? ? shape.shape : nil
              else
                raise "Provided a prefix or suffix of None: #{prefix} and #{suffix}" if p.nil? || s.nil?

                TensorStream.concat([p, s], 0)
              end
      shape
    end

    ##
    # Create tensors of zeros based on state_size, batch_size, and dtype.
    def _zero_state_tensors(state_size, batch_size, dtype)
      get_state_shape = lambda { |s|
        c = _concat(batch_size, s)
        size = TensorStream.zeros(c, dtype: dtype)
        c_static = _concat(batch_size, s, static: true)
        size.set_shape(c_static)
        size
      }

      map_structure(get_state_shape, state_size)
    end
  end
end