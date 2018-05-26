module TensorStream
  module StringHelper
    def camelize(string, uppercase_first_letter = true)
      string = if uppercase_first_letter
                 string.sub(/^[a-z\d]*/) { $&.capitalize }
               else
                 string.sub(/^(?:(?=\b|[A-Z_])|\w)/) { $&.downcase }
               end
      string.gsub(/(?:_|(\/))([a-z\d]*)/) { "#{$1}#{$2.capitalize}" }.gsub('/', '::')
    end
  end
end