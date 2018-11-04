module TensorStream
  ## Collection of machine learning related ops
  module RandomOps
    def RandomOps.included(klass)
      klass.class_eval do
        register_op :glorot_uniform, no_eval: true do |_context, tensor, _inputs|
          seed = tensor.options[:seed]
          random = _get_randomizer(tensor, seed)

          shape = tensor.options[:shape] || tensor.shape.shape
          fan_in, fan_out = if shape.size.zero?
                              [1, 1]
                            elsif shape.size == 1
                              [1, shape[0]]
                            else
                              [shape[0], shape.last]
                            end

          limit = Math.sqrt(6.0 / (fan_in + fan_out))

          minval = -limit
          maxval = limit

          generator = -> { random.rand * (maxval - minval) + minval }
          generate_vector(shape, generator: generator)
        end

        register_op :random_uniform, no_eval: true do |_context, tensor, inputs|
          maxval = tensor.options.fetch(:maxval, 1)
          minval = tensor.options.fetch(:minval, 0)
          seed = tensor.options[:seed]

          random = _get_randomizer(tensor, seed)
          generator = -> { random.rand * (maxval - minval) + minval }
          shape = inputs[0] || tensor.shape.shape
          generate_vector(shape, generator: generator)
        end

        register_op :random_standard_normal, no_eval: true do |_context, tensor, inputs|
          seed = tensor.options[:seed]
          random = _get_randomizer(tensor, seed)
          r = RandomGaussian.new(tensor.options.fetch(:mean), tensor.options.fetch(:stddev), -> { random.rand })
          random = _get_randomizer(tensor, seed)
          generator = -> { r.rand }
          shape = inputs[0] || tensor.shape.shape
          generate_vector(shape, generator: generator)
        end
      end
    end
  end
end