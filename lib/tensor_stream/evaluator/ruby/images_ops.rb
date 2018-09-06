require 'chunky_png'

module TensorStream
  module ImagesOps
    def ImagesOps.included(klass)
      klass.class_eval do
        register_op :decode_png do |_context, tensor, inputs|
          content = inputs[0]
          channels = tensor.options[:channels]
          channels = 4 if channels.zero?

          image = ChunkyPNG::Image.from_blob(content)

          image.grayscale! if channels == 1
          image_data = image.pixels.collect do |pixel|
            color_values = if channels == 4
              [ ChunkyPNG::Color.r(pixel),
                ChunkyPNG::Color.g(pixel),
                ChunkyPNG::Color.b(pixel),
                ChunkyPNG::Color.a(pixel) ]
            elsif channels == 3
              [ ChunkyPNG::Color.r(pixel),
                ChunkyPNG::Color.g(pixel),
                ChunkyPNG::Color.b(pixel)]
            elsif channels == 1
              [ ChunkyPNG::Color.r(pixel) ]
            else
              raise "Invalid channel value #{channels}"
            end
            
            if fp_type?(tensor.data_type)
              color_values.map! { |v| v.to_f }
            end

            color_values
          end
          TensorShape.reshape(image_data.flatten, [image.height, image.width, channels])
        end

        register_op :encode_png do |_context, tensor, inputs|
          image_data = inputs[0]
          height, width, channels = shape_eval(image_data)

          png = ChunkyPNG::Image.new(width, height)
          image_data.each_with_index do |rows, h_index|
            rows.each_with_index do |p_data, w_index|
              if channels == 4
                png[w_index, h_index] = ChunkyPNG::Color.rgba(p_data[0], p_data[1], p_data[2], p_data[3])
              elsif channels == 3
                png[w_index, h_index] = ChunkyPNG::Color.rgb(p_data[0], p_data[1], p_data[2])
              elsif channels == 1
                png[w_index, h_index] = ChunkyPNG::Color.rgb(p_data[0], p_data[0], p_data[0])
              end
            end
          end
          png.to_s
        end
      end
    end
  end
end