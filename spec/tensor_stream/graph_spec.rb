require "spec_helper"
require 'benchmark'
require 'matrix'
# require 'tensor_stream/evaluator/opencl/opencl_evaluator'

RSpec.describe TensorStream::Graph do
  let(:ts) { TensorStream }

  before(:each) do
    TensorStream::Tensor.reset_counters
    TensorStream::Operation.reset_counters
  end

  describe ".named_scope" do
    it "creates named scopes" do
      ts.graph.as_default do |g|
        c = ts.constant(5.0, name: "c")
        expect(c.name).to eq("c")
        c_1 = ts.constant(6.0, name: "c")
        expect(c_1.name).to eq("c_1")

        g.name_scope("nested") do |scope|
          nested_c = ts.constant(10.0, name: "c")
          expect(nested_c.name).to eq("nested/c")

          g.name_scope("inner") do
            nested_inner_c = ts.constant(20.0, name: "c")
            expect(nested_inner_c.name).to eq("nested/inner/c")
            expect(g.get_name_scope).to eq("nested/inner")
          end

          nested_d = ts.constant(10.0, name: "d")
          expect(nested_d.name).to eq("nested/d")
        end
      end
    end
  end

  xdescribe ".parse_from_string" do
    specify do
      pbtext = File.read(File.join('spec','fixtures','linear_regression.pbtxt'))
      graph = TensorStream::Graph.parse_from_string(pbtext)
      # expect(graph.as_graph_def).to eq(pbtext)
      expect(graph.nodes.keys.size).to eq(127)
      expect(graph.nodes.keys.sort).to eq([
        "Add",
        "GradientDescent",
        "GradientDescent/learning_rate",
        "GradientDescent/update_bias/ApplyGradientDescent",
        "GradientDescent/update_weight/ApplyGradientDescent",
        "Mul",
        "Placeholder",
        "Placeholder_1",
        "Pow",
        "Pow/y",
        "Rank",
        "Sum",
        "bias",
        "bias/Assign",
        "bias/initial_value",
        "bias/read",
        "gradients/Add_grad/BroadcastGradientArgs",
        "gradients/Add_grad/Reshape",
        "gradients/Add_grad/Reshape_1",
        "gradients/Add_grad/Shape",
        "gradients/Add_grad/Shape_1",
        "gradients/Add_grad/Sum",
        "gradients/Add_grad/Sum_1",
        "gradients/Add_grad/tuple/control_dependency",
        "gradients/Add_grad/tuple/control_dependency_1",
        "gradients/Add_grad/tuple/group_deps",
        "gradients/Fill",
        "gradients/Mul_grad/BroadcastGradientArgs",
        "gradients/Mul_grad/Mul",
        "gradients/Mul_grad/Mul_1",
        "gradients/Mul_grad/Reshape",
        "gradients/Mul_grad/Reshape_1",
        "gradients/Mul_grad/Shape",
        "gradients/Mul_grad/Shape_1",
        "gradients/Mul_grad/Sum",
        "gradients/Mul_grad/Sum_1",
        "gradients/Mul_grad/tuple/control_dependency",
        "gradients/Mul_grad/tuple/control_dependency_1",
        "gradients/Mul_grad/tuple/group_deps",
        "gradients/Pow_grad/BroadcastGradientArgs",
        "gradients/Pow_grad/Greater",
        "gradients/Pow_grad/Greater/y",
        "gradients/Pow_grad/Log",
        "gradients/Pow_grad/Pow",
        "gradients/Pow_grad/Reshape",
        "gradients/Pow_grad/Reshape_1",
        "gradients/Pow_grad/Select",
        "gradients/Pow_grad/Shape",
        "gradients/Pow_grad/Shape_1",
        "gradients/Pow_grad/Sum",
        "gradients/Pow_grad/Sum_1",
        "gradients/Pow_grad/mul",
        "gradients/Pow_grad/mul_1",
        "gradients/Pow_grad/mul_2",
        "gradients/Pow_grad/mul_3",
        "gradients/Pow_grad/sub",
        "gradients/Pow_grad/sub/y",
        "gradients/Pow_grad/tuple/control_dependency",
        "gradients/Pow_grad/tuple/control_dependency_1",
        "gradients/Pow_grad/tuple/group_deps",
        "gradients/Pow_grad/zeros_like",
        "gradients/Shape",
        "gradients/Sum_grad/DynamicStitch",
        "gradients/Sum_grad/Fill",
        "gradients/Sum_grad/Fill/value",
        "gradients/Sum_grad/Maximum",
        "gradients/Sum_grad/Maximum/y",
        "gradients/Sum_grad/Reshape",
        "gradients/Sum_grad/Shape",
        "gradients/Sum_grad/Shape_1",
        "gradients/Sum_grad/Size",
        "gradients/Sum_grad/Tile",
        "gradients/Sum_grad/add",
        "gradients/Sum_grad/floordiv",
        "gradients/Sum_grad/mod",
        "gradients/Sum_grad/range",
        "gradients/Sum_grad/range/delta",
        "gradients/Sum_grad/range/start",
        "gradients/grad_ys_0",
        "gradients/sub_grad/BroadcastGradientArgs",
        "gradients/sub_grad/Neg",
        "gradients/sub_grad/Reshape",
        "gradients/sub_grad/Reshape_1",
        "gradients/sub_grad/Shape",
        "gradients/sub_grad/Shape_1",
        "gradients/sub_grad/Sum",
        "gradients/sub_grad/Sum_1",
        "gradients/sub_grad/tuple/control_dependency",
        "gradients/sub_grad/tuple/control_dependency_1",
        "gradients/sub_grad/tuple/group_deps",
        "gradients/truediv_grad/BroadcastGradientArgs",
        "gradients/truediv_grad/Neg",
        "gradients/truediv_grad/RealDiv",
        "gradients/truediv_grad/RealDiv_1",
        "gradients/truediv_grad/RealDiv_2",
        "gradients/truediv_grad/Reshape",
        "gradients/truediv_grad/Reshape_1",
        "gradients/truediv_grad/Shape",
        "gradients/truediv_grad/Shape_1",
        "gradients/truediv_grad/Sum",
        "gradients/truediv_grad/Sum_1",
        "gradients/truediv_grad/mul",
        "gradients/truediv_grad/tuple/control_dependency",
        "gradients/truediv_grad/tuple/control_dependency_1",
        "gradients/truediv_grad/tuple/group_deps",
        "init",
        "range",
        "range/delta",
        "range/start",
        "save/Assign",
        "save/Assign_1",
        "save/Const",
        "save/RestoreV2",
        "save/RestoreV2/shape_and_slices",
        "save/RestoreV2/tensor_names",
        "save/SaveV2",
        "save/SaveV2/shape_and_slices",
        "save/SaveV2/tensor_names",
        "save/control_dependency",
        "save/restore_all",
        "sub",
        "truediv",
        "truediv/y",
        "weight",
        "weight/Assign",
        "weight/initial_value",
        "weight/read"]
      )
    end

    xspecify "reload simple operation" do
      pbtext = File.read(File.join('spec','fixtures','matmul_graph.pbtxt'))
      graph = TensorStream::Graph.parse_from_string(pbtext)
      tensor = graph.get_tensor_by_name("tanh_2:0")
      graph.as_default do
        sess = ts.session
        expect(tr(sess.run(tensor))).to eq([[1.0, 1.0], [1.0, 1.0]])
      end
    end

    xspecify "reload gradient operation" do
      pbtext = File.read(File.join('spec','fixtures','gradients.pbtxt'))
      graph = TensorStream::Graph.parse_from_string(pbtext)
      tensor_2 = graph.get_tensor_by_name("gradients/add_grad/Reshape_1")
      tensor_1 =  graph.get_tensor_by_name("gradients/add_grad/Reshape")
      graph.as_default do
        sess = ts.session
        expect(tr(sess.run(tensor_1))).to eq([[1, 1], [1, 1], [1, 1]])
        expect(tr(sess.run(tensor_2))).to eq(6)
      end
    end

    xspecify "complex" do
      pbtext = File.read(File.join('spec','fixtures','neural_network.pbtxt'))
      graph = TensorStream::Graph.parse_from_string(pbtext)

    end
  end
end