require 'tensor_stream/utils/py_ports'
##
# This is ruby port of https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/embedding_ops.py credited to TensorFlow Authors.
#
# This is a best effort translation and certain things need to be changed in order to work with ruby conventions
#
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
            raise "not yet supported!"
          else
            raise TensorStream::ValueError, "Unrecognized partition strategy: " + partition_strategy
          end

          p_assignments = TensorStream.cast(p_assignments, :int32)
          gather_ids = TensorStream.dynamic_partition(new_ids, p_assignments, np)
          pindices = TensorStream.dynamic_partition(original_indices, p_assignments, np)
          partitioned_result = []
          (0...np).each do |p|
            pids = gather_ids[p]
            result = nil
            TensorStream.colocate_with(params[p]) do
              result = TensorStream.gather(params[p], pids)
              if transform_fn
                # If transform_fn is provided, the clip_by_norm precedes
                # the transform and hence must be co-located. See below
                # for the counterpart if transform_fn is not proveded.
                result = transform_fn.call(_clip(result, pids, max_norm))
              end
            end
            partitioned_result << result
          end
          ret = TensorStream.dynamic_stitch(pindices, partitioned_result, name: name)

          if transform_fn.nil?
            element_shape_s = params[0].shape[1..-1]
            params[1..-1].each { |p| element_shape_s = element_shape_s.merge_with(p.shape[1..-1]) }
          else
            element_shape_s = ret.shape[1..-1]
          end

           # Compute the dynamic element shape.
          element_shape_d = if element_shape_s.fully_defined?
                               element_shape_s
                            elsif transform_fn.nil?
                              # It's important that we compute params[0].shape on the right device
                              # to avoid data motion.
                              TensorStream.colocate_with(params[0]) do
                                params_shape = TensorStream.shape(params[0])
                                params_shape[1..-1]
                              end
                            else
                              TensorStream.shape(ret)[1..-1]
                            end
          ret = TensorStream.reshape(ret, TensorStream.concat([TensorStream.shape(ids), element_shape_d], 0))
          ret = _clip(ret, ids, max_norm) unless transform_fn
          ret
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