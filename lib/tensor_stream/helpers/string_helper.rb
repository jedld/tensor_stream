module TensorStream
  # helper string methods usually found in ActiveSupport but
  # need to replicate here since we don't want to use ActiveSupport
  module StringHelper
    def camelize(string, uppercase_first_letter = true)
      string = if uppercase_first_letter
                 string.sub(/^[a-z\d]*/) { $&.capitalize }
               else
                 string.sub(/^(?:(?=\b|[A-Z_])|\w)/) { $&.downcase }
               end
      string.gsub(/(?:_|(\/))([a-z\d]*)/) { "#{$1}#{$2.capitalize}" }.gsub("/", "::")
    end

    def underscore(string)
      string.gsub(/::/, "/").
        gsub(/([A-Z]+)([A-Z][a-z])/, '\1_\2').
        gsub(/([a-z\d])([A-Z])/, '\1_\2').
        tr("-", "_").downcase
    end

    def symbolize_keys(hash)
      hash.map { |k, v|
        [k.to_sym, v]
      }.to_h
    end

    def constantize(camel_cased_word)
      names = camel_cased_word.split("::")

      # Trigger a built-in NameError exception including the ill-formed constant in the message.
      Object.const_get(camel_cased_word) if names.empty?

      # Remove the first blank element in case of '::ClassName' notation.
      names.shift if names.size > 1 && names.first.empty?

      names.inject(Object) do |constant, name|
        if constant == Object
          constant.const_get(name)
        else
          candidate = constant.const_get(name)
          next candidate if constant.const_defined?(name, false)
          next candidate unless Object.const_defined?(name)

          # Go down the ancestors to check if it is owned directly. The check
          # stops when we reach Object or the end of ancestors tree.
          constant = constant.ancestors.inject { |const, ancestor|
            break const    if ancestor == Object
            break ancestor if ancestor.const_defined?(name, false)
            const
          }

          # owner is in Object, so raise
          constant.const_get(name, false)
        end
      end
    end
  end
end
