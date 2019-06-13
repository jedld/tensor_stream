require "chunky_png"


module TensorStream
  module ImagesOps
    def self.included(klass)
      klass.class_eval do
        register_op :decode_png do |_context, tensor, inputs|
          content = inputs[0]
          channels = tensor.options[:channels]
          resample_new_shape = tensor.options[:new_shape]
          resample_method = tensor.options[:resample_method] || :bilinear
          channels = 4 if channels.zero?

          image = ChunkyPNG::Image.from_blob(content)

          image.grayscale! if channels == 1

          if resample_new_shape
            case resample_method
            when :bilinear
              image.resample_bilinear!(resample_new_shape[1], resample_new_shape[0]) # width, # height
            when :nearest_neighbor
              image.resample_nearest_neighbor!(resample_new_shape[1], resample_new_shape[0])
            else
              raise TensorStream::ValueError, "invalid resample method provided #{resample_method}. Available (:bilinear, :nearest_neighbor)"
            end
          end

          image_data = image.pixels.collect { |pixel|
            color_values = if channels == 4
              [ChunkyPNG::Color.r(pixel),
               ChunkyPNG::Color.g(pixel),
               ChunkyPNG::Color.b(pixel),
               ChunkyPNG::Color.a(pixel),]
            elsif channels == 3
              [ChunkyPNG::Color.r(pixel),
               ChunkyPNG::Color.g(pixel),
               ChunkyPNG::Color.b(pixel),]
            elsif channels == 1
              [ChunkyPNG::Color.r(pixel)]
            else
              raise "Invalid channel value #{channels}"
            end

            color_values.map!(&:to_f) if fp_type?(tensor.data_type)

            color_values
          }
          TensorShape.reshape(image_data, [image.height, image.width, channels])
        end

        register_op :decode_jpg do |_context, tensor, inputs|
          require "jpeg"

          content = inputs[0]
          channels = tensor.options[:channels]
          channels = 3 if channels.zero?

          image = Jpeg::Image.open_buffer(content)
          source_channels = image.color_info == :gray ? 1 : 3

          image_data = image.raw_data.map do |pixel|
            if source_channels == channels
              pixel
            elsif source_channels = 1 && channels == 3
              [pixel, pixel, pixel]
            elsif source_channels = 3 && channels == 1
              raise TensorStream::ValueError, "color to grayscale not supported for jpg"
            end
          end.flatten

          image_data.map!(&:to_f) if fp_type?(tensor.data_type)

          TensorShape.reshape(image_data, [image.height, image.width, channels])
        end

        register_op :encode_png do |_context, tensor, inputs|
          image_data = inputs[0]
          height, width, channels = shape_eval(image_data)

          resample_new_shape = tensor.options[:new_shape]
          resample_method = tensor.options[:resample_method] || :bilinear

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

          if resample_new_shape
            case resample_method
            when :bilinear
              png.resample_bilinear!(resample_new_shape[1], resample_new_shape[0]) # width, # height
            when :nearest_neighbor
              png.resample_nearest_neighbor!(resample_new_shape[1], resample_new_shape[0])
            else
              raise TensorStream::ValueError, "invalid resample method provided #{resample_method}. Available (:bilinear, :nearest_neighbor)"
            end
          end

          png.to_s
        end
      end
    end
  end
end
