module TensorStream
  class Graphml < Serializer
    def initialize
    end

    def get_string(tensor, session = nil)
      tensor = TensorStream.convert_to_tensor(tensor) unless tensor.is_a?(Tensor)
      @session = session
      @name = tensor.name
      @last_session_context = session ? session.last_session_context : {}
      groups = {}

      arr_buf = []
      arr_buf << '<?xml version="1.0" encoding="UTF-8"?>'
      arr_buf << '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:y="http://www.yworks.com/xml/graphml"
      xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">'
      arr_buf << '<key id="d0" for="node" attr.name="label" attr.type="string"/>'
      arr_buf << '<key id="d1" for="node" attr.name="formula" attr.type="string"/>'
      arr_buf << '<key id="d2" for="node" attr.name="color" attr.type="string"/>'
      arr_buf << '<key id="d3" for="node" attr.name="value" attr.type="string"/>'
      arr_buf << '<key attr.name="description" attr.type="string" for="edge" id="d12"/>'
      arr_buf << '<key for="edge" id="d13" yfiles.type="edgegraphics"/>'
      arr_buf << '<key for="node" id="d9" yfiles.type="nodegraphics"/>'
      arr_buf << "<graph id=\"g_#{_gml_string(tensor.name)}\" edgedefault=\"directed\">"
      arr_buf << "<node id=\"out\">"
      arr_buf << "<data key=\"d0\">out</data>"
      arr_buf << "<data key=\"d2\">red</data>"
      arr_buf << "<data key=\"d9\">"
      arr_buf << "<y:ShapeNode>"
      arr_buf << "  <y:Fill color=\"#FF0000\" transparent=\"false\"/>"
      arr_buf << "  <y:NodeLabel alignment=\"center\">out</y:NodeLabel>"
      arr_buf << "</y:ShapeNode>"
      arr_buf << "</data>"
      arr_buf << "</node>"

      to_graph_ml(tensor, arr_buf, {}, groups)
      #dump groups
      groups.each do |k, g|
        arr_buf << create_group(k, k, g)
      end

      output_edge(tensor, "out", arr_buf)
      arr_buf << "</graph>"
      arr_buf << "</graphml>"
      arr_buf.flatten.join("\n")
    end

    private

    def add_to_group(groups, name, arr_buf)
      name_parts = name.split('/')
      return false if name_parts.size < 2

      prefix = name_parts.shift

      ptr = find_or_create_group(prefix, groups)

      Kernel.loop do
        next_group = ptr[:group]
        ptr = find_or_create_group(prefix, next_group)
        break if name_parts.size < 2
        prefix = name_parts.shift
      end

      ptr[:buf] << arr_buf
      true
    end

    def find_or_create_group(prefix, groups)
      if !groups[prefix]
        groups[prefix] = { buf: [], group: {} }
      end

      return groups[prefix]
    end

    def create_group(id, title, group)
      arr_buf = []
      arr_buf << "<node id=\"#{id}\" yfiles.foldertype=\"group\">"
      arr_buf << '<data key="d9">'
      arr_buf << '<y:ProxyAutoBoundsNode>'
      arr_buf << '<y:Realizers active="0">'
      arr_buf << '<y:GroupNode>'
      arr_buf << '<y:Fill color="#CAECFF84" transparent="false"/>'
      arr_buf << '<y:BorderStyle color="#666699" type="dotted" width="1.0"/>'
      arr_buf << '<y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#99CCFF" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="67.18603515625" x="-8.593017578125" y="0.0">'+ title + '</y:NodeLabel>'
      arr_buf << '<y:Shape type="roundrectangle"/>'
      arr_buf << '</y:GroupNode>'
      arr_buf << '</y:Realizers>'
      arr_buf << '</y:ProxyAutoBoundsNode>'
      arr_buf << '</data>'
      arr_buf << '<graph edgedefault="directed" id="n105:">'
      arr_buf << group[:buf]
      group[:group].each do |k, g|
        arr_buf << create_group(k, k, g)
      end
      arr_buf << '</graph>'
      arr_buf << '</node>'
      arr_buf
    end

    def _val(tensor)
      # JSON.pretty_generate(@last_session_context[tensor.name])
      @last_session_context[tensor.name] || @last_session_context[:_cache][tensor.name]
    end

    def to_graph_ml(tensor, arr_buf = [], added = {}, groups = {}, _id = 0)
      puts tensor.name
      return unless tensor.is_a?(Operation)

      added[tensor.name] = true
      node_buf = []
      node_buf << "<node id=\"#{_gml_string(tensor.name)}\">"
      node_buf << "<data key=\"d0\">#{tensor.operation}</data>"
      node_buf << "<data key=\"d1\">#{tensor.to_math(true, 1)}</data>"
      node_buf << "<data key=\"d2\">blue</data>"

      if @last_session_context[tensor.name]
        arr_buf << "<data key=\"d3\">#{_val(tensor)}</data>"
      end
      node_buf << "<data key=\"d9\">"
      node_buf << "<y:ShapeNode>"
      if tensor.internal?
        node_buf << "  <y:Fill color=\"#FFFF99\" transparent=\"false\"/>"
      else
        node_buf << "  <y:Fill color=\"#99CC00\" transparent=\"false\"/>"
      end
      node_buf << "  <y:NodeLabel alignment=\"center\">#{tensor.operation}</y:NodeLabel>"
      node_buf << "</y:ShapeNode>"
      node_buf << "</data>"
      node_buf << "</node>"

      if !add_to_group(groups, tensor.name, node_buf)
        add_to_group(groups, "program/#{tensor.name}", node_buf)
      end

      tensor.inputs.each do |input|
        next unless input
        next if added[input.name]

        next to_graph_ml(input, arr_buf, added, groups) if input.is_a?(Operation)

        added[input.name] = true
        input_buf = []
        if input.is_a?(Variable)
          input_buf << "<node id=\"#{_gml_string(input.name)}\">"
          input_buf << "<data key=\"d0\">#{input.name}</data>"
          input_buf << "<data key=\"d2\">green</data>"
          if @last_session_context[input.name]
            input_buf << "<data key=\"d3\">#{_val(tensor)}</data>"
          end
          input_buf << "<data key=\"d9\">"
          input_buf << "<y:ShapeNode>"
          input_buf << "  <y:Fill color=\"#33CCCC\" transparent=\"false\"/>"
          input_buf << "  <y:NodeLabel alignment=\"center\">#{input.name}</y:NodeLabel>"
          input_buf << "</y:ShapeNode>"
          input_buf << "</data>"
          input_buf << "</node>"
        elsif input.is_a?(Placeholder)
          input_buf << "<node id=\"#{_gml_string(input.name)}\">"
          input_buf << "<data key=\"d9\">"
          input_buf << "<y:ShapeNode>"
          input_buf << "  <y:Fill color=\"#FFCC00\" transparent=\"false\"/>"
          input_buf << "  <y:NodeLabel alignment=\"center\">#{input.name}</y:NodeLabel>"
          input_buf << "</y:ShapeNode>"
          input_buf << "</data>"
          if @last_session_context[input.name]
            input_buf << "<data key=\"d3\">#{_val(tensor)}</data>"
          end
          input_buf << "</node>"
        elsif input.is_a?(Tensor)
          input_buf << "<node id=\"#{_gml_string(input.name)}\">"
          input_buf << "<data key=\"d0\">#{input.name}</data>"
          input_buf << "<data key=\"d2\">black</data>"
          input_buf << "<data key=\"d9\">"
          input_buf << "<y:ShapeNode>"

          if input.internal?
            input_buf << "  <y:Fill color=\"#C0C0C0\" transparent=\"false\"/>"
          else
            input_buf << "  <y:Fill color=\"#FFFFFF\" transparent=\"false\"/>"
          end


          input_buf << "  <y:NodeLabel alignment=\"center\">#{input.name}</y:NodeLabel>"

          input_buf << "</y:ShapeNode>"
          input_buf << "</data>"
          input_buf << "</node>"
        end

        if !add_to_group(groups, input.name, input_buf)
          if input.is_a?(Variable)
            add_to_group(groups, "variable/#{input.name}", input_buf)
          else
            add_to_group(groups, "program/#{input.name}", input_buf)
          end
        end
      end

      tensor.inputs.each_with_index do |input, index|
        next unless input
        output_edge(input, tensor, arr_buf, index)
      end
    end

    def _gml_string(str)
      str.gsub('/','-')
    end

    def output_edge(input, tensor, arr_buf, index = 0)
      target_name = tensor.is_a?(Tensor) ? tensor.name : tensor
      arr_buf << "<edge source=\"#{_gml_string(input.name)}\" target=\"#{_gml_string(target_name)}\">"
      arr_buf << "<data key=\"d13\">"

      arr_buf << "<y:PolyLineEdge>"
      arr_buf << "<y:EdgeLabel >"
      if !@last_session_context.empty?
        arr_buf << "<![CDATA[  #{_val(input)}  ]]>"
      else
        if input.shape.shape.nil?
          arr_buf << "<![CDATA[ #{input.data_type.to_s} ? ]]>"
        else
          arr_buf << "<![CDATA[ #{input.data_type.to_s} #{input.shape.shape.empty? ? 'scalar' : input.shape.shape.to_json}  ]]>"
        end
      end
      arr_buf << "</y:EdgeLabel >"
      arr_buf << "<y:Arrows source=\"none\" target=\"standard\"/>"
      if index == 0
        arr_buf << "<y:LineStyle color=\"#FF0000\" type=\"line\" width=\"1.0\"/>"
      else
        arr_buf << "<y:LineStyle color=\"#0000FF\" type=\"line\" width=\"1.0\"/>"
      end
      arr_buf << "</y:PolyLineEdge>"
      arr_buf << "</data>"
      arr_buf << "</edge>"
    end
  end
end