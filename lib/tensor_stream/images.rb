module TensorStream
  module Images
    extend OpHelper

    def self.decode_png(contents, channels: 0, dtype: :uint8, name: nil)
      _op(:decode_png, contents, channels: channels, dtype: dtype, name: name)
    end
  end
end