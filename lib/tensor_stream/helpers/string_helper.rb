module TensorStream
  # helper string methods usually found in ActiveSupport but
  # need to replicate here
  module StringHelper
    def camelize(string, uppercase_first_letter = true)
      string = if uppercase_first_letter
                 string.sub(/^[a-z\d]*/) { $&.capitalize }
               else
                 string.sub(/^(?:(?=\b|[A-Z_])|\w)/) { $&.downcase }
               end
      string.gsub(/(?:_|(\/))([a-z\d]*)/) { "#{$1}#{$2.capitalize}" }.gsub('/', '::')
    end

    def underscore(string)
      string.gsub(/::/, '/').
      gsub(/([A-Z]+)([A-Z][a-z])/,'\1_\2').
      gsub(/([a-z\d])([A-Z])/,'\1_\2').
      tr("-", "_").
      downcase
    end

    def symbolize_keys(hash)
      hash.map do |k, v|
        [k.to_sym, v]
      end.to_h
    end
  end
end