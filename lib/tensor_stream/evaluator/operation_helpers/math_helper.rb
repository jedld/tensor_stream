module TensorStream
  # varoius utility functions for array processing
  module MathHelper
    # Calculates value of y = 1.0 / ( 1.0 + exp( -x ) )
    def sigmoid(val)
      1 / (1 + Math.exp(-val))
    end
  end
end