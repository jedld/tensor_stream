module TensorStream
  ## Collection of machine learning related ops
  module RandomOps
    def self.included(klass)
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

        register_op :truncated_normal, no_eval: true do |_context, tensor, inputs|
          seed = tensor.options[:seed]
          random = _get_randomizer(tensor, seed)
          r = RandomGaussian.new(tensor.options.fetch(:mean), tensor.options.fetch(:stddev), -> { random.rand })

          generator = -> { r.rand }
          shape = inputs[0] || tensor.shape.shape
          random_values = Array.new(shape.reduce(:*) || 1) {
            generator.call
          }
          mean = random_values.reduce(:+) / random_values.size

          # standard deviation

          stddev = Math.sqrt(random_values.map { |v| (v - mean)**2 }.reduce(:+) / (random_values.size - 1))
          minval = random_values.min
          maxval = random_values.max
          max_iterations = 100

          if (minval.infinite? && minval < 0.0) || (maxval < mean)
            # Reverse all calculations. normMin and normMax will be flipped.
            a = minval
            minval = maxval
            maxval = a
            stddev = -stddev
          end

          norm_min = (minval - mean) / stddev
          norm_max = (maxval - mean) / stddev

          val = random_values.map { |v|
            iterations = 0
            pick = v
            while (pick > norm_max) || (pick < norm_min)
              pick = generator.call
              iterations += 1
              if iterations > max_iterations
                pick = v
                break
              end
            end

            pick
          }

          TensorShape.reshape(val, shape)
        end
      end
    end
  end
end
