require 'ostruct'

module TensorStream
  # Convenience class for specifying valid data_types
  module Types
    def self.int16
      :int16
    end

    def self.uint16
      :uint16
    end

    def self.float16
      :float16
    end

    def self.float32
      :float32
    end

    def self.int32
      :int32
    end

    def self.uint32
      :uint32
    end

    def self.uint8
      :uint8
    end

    def self.float64
      :float64
    end

    def self.string
      :string
    end

    def self.boolean
      :boolean
    end
  end
end
