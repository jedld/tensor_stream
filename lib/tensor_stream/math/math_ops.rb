module TensorStream
  # High level math functions
  class Maths
    extend TensorStream::OpHelper

    module MathFunctions

      ##
      # Normalizes along dimension axis using an L2 norm.
      def l2_normalize(x, axis: nil, epsilon: 1e-12, name: nil)
        TensorStream.name_scope(name, "l2_normalize", values: [x]) do |name|
          x = TensorStream.convert_to_tensor(x, name: "x")
          square_sum = TensorStream.reduce_sum(TensorStream.square(x), axis, keepdims: true)
          x_inv_norm = TensorStream.rsqrt(TensorStream.maximum(square_sum, epsilon))
          TensorStream.multiply(x, x_inv_norm, name: name)
        end
      end
    end

    extend MathFunctions
  end
end