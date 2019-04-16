module TensorStream
  class Trackable
    attr_accessor :update_uid, :unconditional_checkpoint_dependencies, :unconditional_dependency_names, :name_based_restores

    def set_attr_tracking
      @attr_tracking ||= true
      @attr_tracking
    end

    def attr_tracking=(value)
      @attr_tracking = value
    end

    def maybe_initialize_trackable
      return if @unconditional_checkpoint_dependencies

      @unconditional_checkpoint_dependencies = []
      @unconditional_dependency_names = {}
      @unconditional_deferred_dependencies = {}

      raise "Internal error: the object had an update UID set before its " +
        "initialization code was run." if @update_uid

      @update_uid = -1
      @name_based_restores = Set.new
    end

    def no_dependency(value)
      value
    end

    def name_based_attribute_restore(checkpoint)
      @name_based_restores << checkpoint
      if @update_uid < checkpoint.restore_uid
        checkpoint.eager_restore(self)
        @update_uid = checkpoint.restore_uid
      end
    end

    def checkpoint_dependencies
      @unconditional_checkpoint_dependencies
    end

    def unconditional_deferred_dependencies
      @unconditional_deferred_dependencies
    end

    def lookup_dependency(name)
      @unconditional_dependency_names.fetch(name, nil)
    end

    def add_variable_with_custom_getter(
      kwargs_for_getter = {}, name, shape: nil, dtype: :float32,
      initializer: nil, getter: nil, overwrite: false)
      maybe_initialize_trackable
      new_variable = getter(kwargs_for_getter, name: name, shape: shape, dtype: dtype, initializer: initializer)
      if !overwrite || new_variable.is_a?(Trackable)
        track_trackable(new_variable, name: name, overwrite: overwrite)
      else
        new_variable
      end
    end
  end

  ##
  # Manages dependencies on other objects.
  class AutoTrackable < Trackable
  end
end