require 'json'

module TensorStream
  module Train
    # High level class used for loading and saving variables
    class Saver
      include TensorStream::OpHelper

      def save(session, outputfile, global_step: nil,
               latest_filename: nil,
               meta_graph_suffix: 'meta',
               write_meta_graph: true,
               write_state: true,
               strip_default_attrs: false)
        vars = TensorStream::Graph.get_default_graph.get_collection(GraphKeys::GLOBAL_VARIABLES)

        variables = {}
        graph = {}
        gs =  eval_global_step(session, global_step)
        output_dump = {
          variables: variables,
          graph: graph,
          global_step: gs
        }

        vars.each do |variable|
          variables[variable.name] = variable.read_value
        end

        basename = File.basename(outputfile)
        path = File.dirname(outputfile)

        new_filename = File.join(path, [basename, gs].compact.join('-'))
        File.write(new_filename, output_dump.to_json)

        path
      end

      def restore(_session, inputfile)
        input_dump = JSON.parse(File.read(inputfile))

        vars = TensorStream::Graph.get_default_graph.get_collection(GraphKeys::GLOBAL_VARIABLES)
        vars.each do |variable|
          variable.value = input_dump['variables'][variable.name]
        end
      end

      private

      def build_internal(names_to_saveables, reshape: false, sharded: false, max_to_keep: 5,
        keep_checkpoint_every_n_hours: 10000.0,
        name: nil,
        restore_sequentially: false,
        filename: "model",
        build_save: true,
        build_restore: true)
        saveables = _validate_and_slice_inputs(names_to_saveables)
      end

      def _validate_and_slice_inputs(names_to_saveables)
        saveables = []
        seen_ops = []

        names_to_saveables.values.sort_by { |item| item[0] }.each do |name, op|
          _saveable_objects_for_op(op, name).each do |converted_saveable_object|
            _add_saveable(saveables, seen_ops, converted_saveable_object)
          end
        end
        saveables
      end

      def _add_saveable(saveables, seen_ops, saveable)
        raise TensorStreamm::ValueError, "The same saveable will be restored with two names: #{saveable.name}" if seen_ops.include?(saveable.op)
        saveables << saveable
        seen_ops << saveable.op
      end

      def save_op(filename_tensor, saveables)
        tensor_names = []
        tensors = []
        tensor_slices = []

        saveables.each do |saveable|
          saveable.specs.each do |spec|
            tensor_names << spec.name
            tensors << spec.tensor
            tensor_slices << spec.slice_spec
          end
        end
        i_op(:save_v2, filename_tensor, tensor_names, tensor_slices, tensors)
      end

      def eval_global_step(session, global_step)
        return nil if global_step.nil?

        if global_step.is_a?(Tensor)
          session.last_session_context(global_step.name)
        elsif global_step.is_a?(String) || global_step.is_a?(Symbol)
          session.last_session_context(global_step)
        else
          global_step.to_i
        end
      end
    end
  end
end
