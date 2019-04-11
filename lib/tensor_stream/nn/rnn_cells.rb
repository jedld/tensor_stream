##
# This meant a ruby port of tensorflow/python/ops/rnn_cell_impl.py credited to TensorFlow Authors.
# The original python source file this is based on falls under the Apache License, Version 2.0
#
# However it is not expected that translation is not perfect as there are changes needed in order to follow
# ruby conventions
module TensorStream
  class RNNCell
    include TensorStream::TensorUtils

    protected

    ##
    # Concat that enables int, Tensor, or TensorShape values.
    # This function takes a size specification, which can be an integer, a
    # TensorShape, or a Tensor, and converts it into a concatenated Tensor
    # (if static = False) or a list of integers (if static = True).
    # Args:
    # prefix: The prefix; usually the batch size (and/or time step size).
    #  (TensorShape, int, or Tensor.)
    # suffix: TensorShape, int, or Tensor.
    # static: If `True`, return a python list with possibly unknown dimensions.
    #  Otherwise return a `Tensor`.
    # Returns:
    #   shape: the concatenation of prefix and suffix.
    # Raises:
    # ValueError: if `suffix` is not a scalar or vector (or TensorShape).
    # ValueError: if prefix or suffix was `None` and asked for dynamic
    #      Tensors out.
    def concat(prefix, suffix, static: false)
    #   if prefix.is_a?(Tensor)
    #     p = prefix
    #     p_static = constant_value(prefix)
    #     if p.shape.ndims == 0:
    #   p = array_ops.expand_dims(p, 0)
    # elif p.shape.ndims != 1:
    #   raise ValueError("prefix tensor must be either a scalar or vector, "
    #                    "but saw tensor: %s" % p)
    #   end
    end
  end
end