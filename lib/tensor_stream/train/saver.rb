require 'json'

module TensorStream
  module Train
    class Saver
      def save(session, outputfile,
                global_step: nil,
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
          variables[variable.name] = variable.value
        end

        basename = File.basename(outputfile)
        path = File.dirname(outputfile)

        new_filename = File.join(path, [basename, gs].compact.join('-'))
        File.write(new_filename, output_dump.to_json)

        path
      end

      def restore(session, inputfile)
        input_dump = JSON.parse(File.read(inputfile))

        vars = TensorStream::Graph.get_default_graph.get_collection(GraphKeys::GLOBAL_VARIABLES)
        vars.each do |variable|
          variable.value = input_dump["variables"][variable.name]
        end
      end

      private

      def eval_global_step(session, global_step)
        return nil if global_step.nil?

        if (global_step.is_a?(Tensor))
          session.last_session_context(global_step.name)
        elsif (global_step.is_a?(String) || global_step.is_a?(Symbol))
          session.last_session_context(global_step)
        else
          global_step.to_i
        end
      end
    end
  end
end