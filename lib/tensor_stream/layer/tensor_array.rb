module TensorStream
  class TensorArray
    include TensorStream::OpHelper

    attr_reader :flow
    attr_accessor :infer_shape, :element_shape

    def initialize(dtype, size: nil, dynamic_size: nil, clear_after_read: nil, tensor_array_name: nil, handle: nil, flow: nil, infer_shape: true, element_shape: nil, name: nil)
      raise TensorStream::ValueError, "Cannot construct with both handle and tensor_array_name" if handle && tensor_array_name
      raise TensorStream::ValueError, "Handle must be a Tensor" if handle && !handle.is_a?(Tensor)
      raise TensorStream::ValueError, "Size must be provided if handle is not provided" if handle.nil? && size.nil?
      raise TensorStream::ValueError, "Cannot provide both a handle and size at the same time" if handle && size
      raise TensorStream::ValueError, "Cannot provide both a handle and element_shape at the same time" if handle && element_shape
      raise TensorStream::ValueError, "Cannot provide both a handle and dynamic_size at the same time" if handle && dynamic_size
      raise TensorStream::ValueError, "Cannot provide both a handle and clear_after_read at the same time" if handle && clear_after_read

      clear_after_read = true unless clear_after_read
      dynamic_size ||= false

      @dtype = dtype

      @element_shape = if element_shape
                        @infer_shape = true
                        [TensorShape.new(element_shape)]
                      else
                        @infer_shape = infer_shape
                        []
                      end
      TensorStream.name_scope(name, "TensorArray", values: [handle, size, flow]) do |scope|
        if handle
          @handle = handle
          raise TensorStream::ValueError, "flow must not be None if handle is not None." unless flow
          @flow = flow
        else
          @handle, @flow = i_op(:tensor_array_v3, dtype: dtype, size: size, element_shape: element_shape, dynamic_size: dynamic_size, clear_after_read: clear_after_read, tensor_array_name: tensor_array_name, name: scope)
        end
      end
    end

    ##
    # Unstack the values of a `Tensor` in the TensorArray.
    def unstack(value, name: nil)
      TensorStream.name_scope(name, "TensorArrayUnstack", values: [@handle, value]) do
        num_elements = TensorStream::shape(value)[0]
        scatter(TensorStream.range(0, num_elements), value, name: name)
      end
    end

    ##
    # Scatter the values of a `Tensor` in specific indices of a `TensorArray`.
    def scatter(indices, value, name: nil)
      TensorStream.name_scope(name, "TensorArrayScatter", values: [value, indices]) do
        value = TensorStream.convert_to_tensor(value, name: 'value')
        flow_out = i_op(:tensor_array_scatter_v3, @handle, indices, value, flow_in: @flow, name: name)
        ta = TensorArray.new(@dtype, handle: @handle, flow: flow_out)
        ta.infer_shape = @infer_shape
        ta.element_shape = @element_shape
        if ta.infer_shape
          val_shape = flow_out.op.inputs[2].shape
          element_shape = TensorShape.unknown_shape
          element_shape = TensorShape.new(val_shape[1..val_shape.ndims]) if val_shape.dims
          ta.merge_element_shape(element_shape)
        end
      end
    end

    ##
    # Changes the element shape of the array given a shape to merge with.
    def merge_element_shape(shape)
      raise TensorStream::ValueError, "Inconsistent shapes: saw #{shape} but expected #{@element_shape[0]} (and infer_shape=True)" if @element_shape && !shape.compatible_with?(@element_shape[0])

      @element_shape << shape
    end

    def read(index, name: nil)
      value = i_op(:tensor_array_read_v3, index, @flow, handle: @handle, name: @name)
      value.set_shape(@element_shape[0].dims) if @element_shape
      value
    end

    def write(index, value, name: nil)
      TensorStream.name_scope(name, "TensorArrayWrite", values: [@handle, index, value]) do
        value = TensorStream.convert_to_tensor(value, dtype: @dtype, name: "value")
        merge_element_shape(value.shape) if @infer_shape
        flow_out = i_op(:tensor_array_write_v3, @handle, index, value, @flow, name: name)
        ta = TensorArray.new(@dtype, handle: @handle, flow: flow_out)
        ta.infer_shape = @infer_shape
        ta.element_shape = @element_shape
        ta
      end
    end

    def size(name: nil)
      i_op(:tensor_array_size_v3, @handle, @flow, name: name)
    end

    def close(name: nil)
      i_op(:tensor_array_close_v3, @handle, name: name)
    end
  end
end