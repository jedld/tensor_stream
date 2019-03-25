require 'tensor_stream/utils/py_ports'

module TensorStream
  module EmbeddingLookup
    include TensorStream::PyPorts

    ##
    # Looks up `ids` in a list of embedding tensors.
    def embedding_lookup(params, ids, partition_strategy: "mod", name: nil, validate_indices: true, max_norm: nil)
      _embedding_lookup_and_transform(params, ids, partition_strategy: partition_strategy, name: name, max_norm: max_norm, transform_fn: nil)
    end

    ##
    # Helper function for embedding_lookup and _compute_sampled_logits.
    def _embedding_lookup_and_transform(params, ids, partition_strategy: "mod", name: nil, max_norm: nil, transform_fn: nil)
      raise TensorStream::ValueError, "Need at least one param" if params.nil?

      params = [params] unless params.is_a?(Array)

      TensorStream.name_scope(name, "embedding_lookup", values: params + [ids]) do |name|
        np = params.size
        ids = TensorStream.convert_to_tensor(ids, name: "ids")
        if (np == 1) && (transform_fn.nil? || (ids.shape.size == 1))
          result = nil
          TensorStream.colocate_with(params[0]) do
            result = _clip(TensorStream.gather(params[0], ids, name: name), ids, max_norm)
            result = transform_fn.call(result) if transform_fn
          end

          return TensorStream.identity(result)
        else
          flat_ids = TensorStream.reshape(ids, [-1])
          original_indices = TensorStream.range(TensorStream.size(flat_ids))

          p_assignments = nil
          new_ids = nil

          if partition_strategy == "mod"
            p_assignments = flat_ids % np
            new_ids = floor_div(flat_ids, np)
          elsif partition_strategy == "div"
          else
            raise TensorStream::ValueError, "Unrecognized partition strategy: " + partition_strategy
          end

          p_assignments = TensorStream.cast(p_assignments, :int32)
          gather_ids = TensorStream.dynamic_partition(new_ids, p_assignments, np)
          pindices = TensorStream.dynamic_partition(original_indices, p_assignments, np)
          partitioned_result = []
          binding.pry
        end
      end
    end

    def _clip(params, ids, max_norm)
      return params if max_norm.nil?

      ids_rank, ids_static = _rank(ids)
      params_rank, params_static = _rank(params)

      TensorStream.clip_by_norm(params, max_norm, axes: ids_static && params_static ? (ids_rank...params_rank).to_a : TensorStream.range(ids_rank, params_rank))
    end

    def _rank(x)
      rank = TensorStream.convert_to_tensor(x).shape.ndims
      if rank
        [rank, false]
      else
        [TensorStream.rank(x), false]
      end
    end
  end
end