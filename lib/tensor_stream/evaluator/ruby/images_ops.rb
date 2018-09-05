require 'chunky_png'

module TensorStream
  module ImagesOps
    def ImagesOps.included(klass)
      klass.class_eval do
        register_op :decode_png do |_context, tensor, inputs|
          filename = inputs[0]
          channels = tensor.options[:channels]
          channels = 4 if channels.zero?

          image = ChunkyPNG::Image.from_file(filename)

          image.grayscale! if channels == 1
          image_data = image.pixels.collect do |pixel|
            if channels == 4
            [ ChunkyPNG::Color.r(pixel),
              ChunkyPNG::Color.g(pixel),
              ChunkyPNG::Color.b(pixel),
              ChunkyPNG::Color.a(pixel) ]
            elsif channels == 3
              [ ChunkyPNG::Color.r(pixel),
                ChunkyPNG::Color.g(pixel),
                ChunkyPNG::Color.b(pixel),
                ChunkyPNG::Color.a(pixel) ]
            elsif channels == 1
              ChunkyPNG::Color.r(pixel)
            else
              raise "Invalid channel value #{channels}"
            end
          end
          TensorShape.reshape(image_data.flatten, [image.height, image.width, channels])
        end
      end
    end
  end
end