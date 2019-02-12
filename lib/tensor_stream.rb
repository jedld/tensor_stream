require "tensor_stream/version"
require "deep_merge"
require "matrix"
require "concurrent"
require "tensor_stream/exceptions"
require "tensor_stream/helpers/op_helper"
require "tensor_stream/helpers/string_helper"
require "tensor_stream/initializer"
require "tensor_stream/graph_keys"
require "tensor_stream/types"
require "tensor_stream/graph_builder"
require "tensor_stream/graph"
require "tensor_stream/device"
require "tensor_stream/session"
require "tensor_stream/tensor_shape"
require "tensor_stream/helpers/tensor_mixins"
require "tensor_stream/tensor"
require "tensor_stream/constant"
require "tensor_stream/variable"
require "tensor_stream/variable_scope"
require "tensor_stream/operation"
require "tensor_stream/placeholder"
require "tensor_stream/control_flow"
require "tensor_stream/dynamic_stitch"
require "tensor_stream/nn/nn_ops"
require "tensor_stream/evaluator/evaluator"
require "tensor_stream/graph_serializers/packer"
require "tensor_stream/graph_serializers/serializer"
require "tensor_stream/graph_deserializers/protobuf"
require "tensor_stream/graph_deserializers/yaml_loader"
require "tensor_stream/graph_serializers/pbtext"
require "tensor_stream/graph_serializers/graphml"
require "tensor_stream/graph_serializers/yaml"
require "tensor_stream/math_gradients"
require "tensor_stream/debugging/debugging"
require "tensor_stream/utils"
require "tensor_stream/train/utils"
require "tensor_stream/images"

require "tensor_stream/profile/report_tool"

# require 'tensor_stream/libraries/layers'
require "tensor_stream/monkey_patches/patch"
require "tensor_stream/monkey_patches/integer"
require "tensor_stream/monkey_patches/float"
require "tensor_stream/monkey_patches/array"
require "tensor_stream/ops"
require "tensor_stream/trainer"
require "tensor_stream/op_maker"

# module that exposes TensorStream top level functions
module TensorStream
  extend TensorStream::OpHelper
  extend TensorStream::Ops
  extend TensorStream::Debugging
  extend TensorStream::Utils

  def self.__version__
    TensorStream::VERSION
  end
end
