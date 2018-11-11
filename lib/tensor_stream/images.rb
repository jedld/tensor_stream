module TensorStream
  module Images
    extend OpHelper
    extend TensorStream::Utils

    def self.decode_png(contents, channels: 0, dtype: :uint8, name: nil, new_shape: new_shape)
      _op(:decode_png, contents, channels: channels, data_type: dtype, name: name, new_shape: new_shape)
    end

    def self.encode_png(contents, compression: -1, name: nil, new_shape: nil, resample_method: nil)
      check_allowed_types(contents, %i[uint8 uint16])
      contents = convert_to_tensor(contents, dtype: :uint16)
      _op(:encode_png, contents, compression: compression, name: name, new_shape: new_shape, resample_method: resample_method)
    end
  end
end