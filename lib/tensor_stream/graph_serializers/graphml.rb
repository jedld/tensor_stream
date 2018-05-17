module TensorStream
  class Graphml
    def initialize
    end

    def serialize(session, tensor, filename)
      @session = session
      @last_session_context = session.last_session_context

      arr_buf = []
      arr_buf << '<?xml version="1.0" encoding="UTF-8"?>'
      arr_buf << '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">'
      arr_buf << '<key id="d0" for="node" attr.name="label" attr.type="string"/>'
      arr_buf << '<key id="d1" for="node" attr.name="formula" attr.type="string"/>'
      arr_buf << '<key id="d2" for="node" attr.name="color" attr.type="string"/>'
      arr_buf << '<key id="d3" for="node" attr.name="value" attr.type="string"/>'
      arr_buf << "<graph id=\"g_#{_gml_string(tensor.name)}\" edgedefault=\"directed\">"
      arr_buf << "<node id=\"out\">"
      arr_buf << "<data key=\"d0\">out</data>"
      arr_buf << "<data key=\"d2\">red</data>"
      arr_buf << "</node>"
      to_graph_ml(tensor, arr_buf)
      arr_buf << "<edge source=\"#{_gml_string(tensor.name)}\" target=\"out\"/>"
      arr_buf << "</graph>"
      arr_buf << "</graphml>"
      File.write(filename, arr_buf.join("\n"))
    end

    private

    def _val(tensor)
      JSON.pretty_generate(@last_session_context[tensor.name])
    end

    def to_graph_ml(tensor, arr_buf = [], added = {}, _id = 0)
      puts tensor.name
      added[tensor.name] = true
      arr_buf << "<node id=\"#{_gml_string(tensor.name)}\">"
      arr_buf << "<data key=\"d0\">#{tensor.operation}</data>"
      arr_buf << "<data key=\"d1\">#{tensor.to_math(true, 1)}</data>"
      arr_buf << "<data key=\"d2\">blue</data>"
      if @last_session_context[tensor.name]
        arr_buf << "<data key=\"d3\">#{_val(tensor)}</data>"
      end
      arr_buf << "</node>"

      tensor.items.each do |item|
        next unless item
        next if _added[item.name]

        next to_graph_ml(item, arr_buf, added) if item.is_a?(Operation)
        added[item.name] = true
        if item.is_a?(Variable)
          arr_buf << "<node id=\"#{_gml_string(item.name)}\">"
          arr_buf << "<data key=\"d0\">#{item.name}</data>"
          arr_buf << "<data key=\"d2\">green</data>"
          if @last_session_context[item.name]
            arr_buf << "<data key=\"d3\">#{_val(tensor)}</data>"
          end
          arr_buf << "</node>"
        elsif item.is_a?(Placeholder)
          arr_buf << "<node id=\"#{_gml_string(item.name)}\">"
          arr_buf << "<data key=\"d0\">#{item.name}</data>"
          arr_buf << "<data key=\"d2\">yellow</data>"
          if @last_session_context[item.name]
            arr_buf << "<data key=\"d3\">#{_val(tensor)}</data>"
          end
          arr_buf << "</node>"
        else
          arr_buf << "<node id=\"#{_gml_string(item.name)}\">"
          arr_buf << "<data key=\"d0\">#{item.name}</data>"
          arr_buf << "<data key=\"d2\">black</data>"
          if @last_session_context[item.name]
            arr_buf << "<data key=\"d3\">#{_val(tensor)}</data>"
          end
          arr_buf << "</node>"
        end
      end

      tensor.items.each do |item|
        next unless item
        arr_buf << "<edge source=\"#{_gml_string(item.name)}\" target=\"#{_gml_string(tensor.name)}\"/>"
      end
    end

    def _gml_string(str)
      str.gsub('/','-')
    end
  end
end