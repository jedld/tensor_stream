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

    ##
    # Declare a dependency on another `Trackable` object.
    def track_trackable(trackable, name, overwrite: false)
      maybe_initialize_trackable

      raise TensorStream::TypeError, "Trackable.track_trackable() passed type #{trackable.class.to_s}, not a " +
                  "Trackable." unless trackable.is_a?(Trackable)

      new_reference = { name: name, ref: trackable }
      current_object = lookup_dependency(name)
      if !current_object.nil? && !current_object.is_a?(Trackable)
        raise TensorStream::ValueError,
          "Called Trackable.track_trackable with name='#{name}', " +
           "but a Trackable with this name is already declared as a " +
           "dependency. Names must be unique (or overwrite=True)." unless overwrite
        @unconditional_checkpoint_dependencies.each_with_index do |elem, index|
          old_name = elem[:name]
          @unconditional_checkpoint_dependencies[index] = new_reference if name == old_name
        end
      elsif current_object.nil?
        @unconditional_checkpoint_dependencies << new_reference
        handle_deferred_dependencies(name, trackable)
      end

      @unconditional_dependency_names[name] = trackable
      trackable
    end

    ##
    # Pop and load any deferred checkpoint restores into `trackable`.
    def handle_deferred_dependencies(name, trackable)
      maybe_initialize_trackable
      trackable.maybe_initialize_trackable
      deferred_dependencies_list = @unconditional_deferred_dependencies.delete(name) || []
      # raise "TODO"
      # deferred_dependencies_list.sort { |restore| restore.checkpoint.restore_uid }.each do |
    end

  end

  ##
  # Manages dependencies on other objects.
  class AutoTrackable < Trackable
  end
end