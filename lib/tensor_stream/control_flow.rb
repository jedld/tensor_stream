require 'tensor_stream/utils/tensor_utils'
require 'tensor_stream/utils/nn_utils'

module TensorStream
  class ControlFlowContext
    attr_reader :outer_context

    include TensorStream::TensorUtils
    include TensorStream::NNUtils
    extend TensorStream::OpHelper
    include TensorStream::OpHelper

    def initialize(values_def = nil, import_sope = nil)
      @outer_context = TensorStream.get_default_graph.control_flow_context
      @context_stack = []
      if values_def
        init_values_from_proto(values_def, import_scope: import_scope)
      else
        @values = Set.new
        @external_values = []
      end
    end

    def grad_state
      raise TensorStream::NotImplementedError, "Abstract method"
    end

    def back_prop
      raise TensorStream::NotImplementedError, "Abstract method"
    end

    def to_proto(export_scope = nil)
      raise "not implemented"
    end

    def add_name(name)
      @values << name
    end

    def enter
      graph = TensorStream.get_default_graph
      @context_stack << graph.control_flow_context
      graph.control_flow_context = self
    end

    def exit
      graph = TensorStream.get_default_graph
      last_context = @context_stack.pop
      graph.control_flow_context = last_context
    end

    def _exit(data, name: nil)
      data = TensorStream.convert_to_tensor(data)
      _op(:exit, data, name)
    end

    def switch(data, pred, dtype: nil, name: nil)
      TensorStream.name_scope(name, "Switch", values: [data, pred]) do |name|
        data = TensorStream.internal_convert_to_tensor_or_indexed_slices(data, dtype: dtype, name: 'data')
        pred = TensorStream.convert_to_tensor(pred, name: 'pred')
        _op(:switch, data, pred, name: name) if data.is_a?(Tensor)
      end
    end

    def exit_result(result)
      if @outer_context
        result.each { |x|
          map_structure(->(x) { @outer_context.add_name(x.name)}, result) }
      end
    end

    def get_while_context
      return @outer_context.get_while_context if @outer_context

      nil
    end

    def in_outer_context?(op)
      op_ctxt = get_output_context(op)
      outer_ctxt = @outer_context
      while outer_ctxt != op_ctxt
        return false if outer_ctxt.nil?

        outer_ctxt = outer_ctxt.outer_context
      end

      true
    end

    def remove_external_control_edges(op)
      while_ctxt = get_while_context

      internal_control_inputs = op.control_inputs
      if while_ctxt
        internal_control_inputs = []
        op.control_inputs.each do |x|
          ctxt = get_output_context(x)
          internal_control_inputs << x if ctxt && ctxt.get_while_context == while_ctxt
        end
      end

      if internal_control_inputs.size != op.control_inputs.size
        op.control_inputs.clear
        op.add_control_inputs(internal_control_inputs)
      end

      internal_control_inputs
    end

    def self._enter(data, frame_name, is_constant: false, parallel_iterations: 10, user_ref: true, use_input_shape: true, name: nil)
      data = TensorStream.convert_to_tensor(data)
      if data.is_a?(Tensor)
        result = i_op(:enter, data, frame_name: frame_name, is_constant: is_constant, parallel_iterations: parallel_iterations, name: name)
        result.set_shape(data.shape) if use_input_shape
        return result
      end

      raise TensorStream::TypeError, "Type #{data} not supported"
    end


    ##
    # Returns the value of an available element of `inputs`.
    def merge(inputs, name: nil)
      raise TensorStream::ValueError, "At least one of the merge inputs is None: #{inputs}" if inputs.detect { |inp| inp.nil? }

      TensorStream.name_scope(name, "Merge", values: inputs) do |name|
        inputs = inputs.map { |inp| TensorStream.internal_convert_to_tensor(inp, as_ref: true) }
        _op(:merge, *inputs, name: name)
      end
    end

    protected

    def get_output_context(op)
    end
    ##
    #
    def init_values_from_proto(values_def, import_scope: nil)
      raise "not implemented"
    end
  end
  # Defines a TensorStream controlflow op
  class WhileContext < ControlFlowContext
    attr_reader :name, :parallel_iterations, :backprop, :swap_memory, :pivot, :loop_exits, :grad_state

    def initialize(parallel_iterations = 10, maximum_iterations: nil, back_prop: true, swap_memory: false, name: 'while_context', grad_state: nil, context_def: nil, import_scope: nil)
      if context_def
        #TODO: does something that deserializes from protobuf?
      else
        super()
        init_from_args(maximum_iterations, parallel_iterations, backprop, swap_memory, name)
      end
      @grad_state = grad_state
    end

    def get_while_context
      self
    end

    def add_value(val)
      result = val
      if !@values.include?(val.name)
        @values << val.name
        grad_ctxt = TensorStream.get_default_graph.control_flow_context
        if grad_ctxt
          grad_ctxt = grad_ctxt.get_while_context
          if grad_ctxt.grad_state
            forward_ctxt = _get_while_context(val.op)
            if loop_exit?(val.op)
              forward_ctxt = forward_ctxt.outer_context
              forward_ctxt = forward_ctxt.get_while_context if forward_ctxt
            end

            if forward_ctxt == grad_ctxt.grad_state.forward_context
              real_val = grad_ctxt.grad_state.get_real_value(val)
              @external_values[val.name] = real_val
              return real_val
            end
          end
        end

        result = @outer_context.add_value(val) if @outer_context

        _enter = nil
        TensorStream.control_dependencies(nil) do
          _enter = enter(result, @name, is_constant: true, parallel_iterations: @parallel_iterations)
          _enter.graph.prevent_feeding(enter)
        end
        fix_control_inputs_and_context([enter])
        @values << enter.name
        @external_values[val.name] = _enter
        result = _enter
      else
        actual_val = @external_values[val.name]
        result = actual+val if actual_val
      end

      result
    end

    def add_op(op)
      if [:shape, :size, :rank].include?(op.operation)
        grad_ctxt = TensorStream.get_default_graph.get_control_flow_context
        if grad_ctxt
          grad_ctxt = grad_ctxt.get_while_context
          if grad_ctxt.grad_state
            op_input_forward_ctxt = _get_while_context(op.inputs[0].op)
            if op_input_forward_ctxt == grad_ctxt.grad_state.forward_context
              op_input_ctxt = op.inputs[0].op.control_flow_context
              op.control_flow_context = op_input_ctxt
              op_input_ctxt.add_op_internal(op)
              return
            end
          end
        end
      end
      add_op_internal(op)
    end

    def add_op_internal(op)
      if !op.inputs || op.inputs.empty?
        control_inputs = remove_external_control_edges(op)

        op.control_input << get_control_pivot.op if !control_inputs || control_inputs.empty?
        @values << op.name
      else
        op.inputs.size.times do |index|
          x = op.inputs[index]
          real_x = add_value(x)
          op.inputs[index] = real_x if real_x != x
        end
        remove_external_control_edges(op)
        maybe_add_control_dependency(op)
        @values << op.name
      end

      if @outer_context || !is_loop_exit(op)
        op.graph.prevent_fetching(op)
        op.graph.prevent_feeding(op)
      end
    end

    def build_loop(pred, body, loop_vars, shape_invariants, return_same_structure)
      original_loop_vars = loop_vars
      loop_vars = map_structure(->(x) { convert_tensorarray_to_flow(x) }, _flatten(loop_vars))
      loop_vars = TensorStream.convert_n_to_tensor_or_indexed_slices(loop_vars)
      original_body_result, exit_vars = begin
                                          enter
                                          TensorStream.get_default_graph.mutation_lock do
                                            _build_loop(pred, body, original_loop_vars, loop_vars, shape_invariants)
                                          end
                                        ensure
                                          self.exit
                                        end
      flat_result = _flatten(original_body_result, expand_composites: true)
      exit_vars_with_tensor_arrays = convert_flows_to_tensorarrays(flat_result, exit_vars)
      packed_exit_vars = pack_sequence_as(original_body_result, exit_vars_with_tensor_arrays)

      return_same_structure ? packed_exit_vars : ( exit_vars.size == 1 ? packed_exit_vars[0] : packed_exit_vars)
    end

    def build_loop2(pred, body, loop_vars, shape_invariants, return_same_structure)
      loop_vars = loop_vars.map do |v|
        TensorRef.new(v)
      end

      body_fn = body.call(*loop_vars)
      cond_fn = pred.call(*loop_vars)

      _op(:loop_frame, body_fn, cond_fn, *loop_vars)
    end

    protected

    def _build_loop(pred, body, original_loop_vars, loop_vars, shape_invariants)
      flat_loop_vars = _flatten(original_loop_vars, expand_composites: true)
      initialize_values(loop_vars)
      real_vars = loop_vars
      real_vars = loop_vars.map { |x| @outer_context.add_value(x) } if @outer_context
      enter_vars = nil
      TensorStream.control_dependencies(nil) do
        enter_vars = real_vars.map { |x| ControlFlowContext._enter(x, @name, is_constant: false, parallel_iterations: @parallel_iterations, use_input_shape: shape_invariants.nil?)}

        enter_vars.each do |x|
          x.graph.prevent_feeding(x)
          @outer_context.add_inner_op(x.op) if @outer_context
        end
      end

      outer_context = @outer_context
      control_pivot = nil
      while outer_context && control_pivot.nil?
        control_pivot = outer_context.control_pivot
        outer_context = outer_context.outer_context
      end

      if control_pivot
        enter_vars.each do |var|
          var.op.control_input << control_pivot.op if is_loop_constant_enter(var.op.inputs[0].op)
        end
      end

      set_shape_invariants(real_vars, enter_vars, shape_invariants)
      fix_control_inputs_and_context(enter_vars)
      initialize_values(enter_vars)
      @loop_enters = enter_vars

      merge_vars = enter_vars.map { |x| merge([x, x])[0] }
      @pivot_for_pred = merge_vars[0]

      #build the graph for pred
      merge_vars_with_tensor_arrays = convert_flows_to_tensorarrays(flat_loop_vars, merge_vars)
      packed_vars = pack_sequence_as(original_loop_vars, merge_vars_with_tensor_arrays)
      c = TensorStream.convert_to_tensor(pred.call(*packed_vars))
      @pivot = _op(:loop_cond, c, name: "LoopCond")
      switch_vars = merge_vars.map { |x| switch_ref_or_tensor(x, @pivot) }

      #build the graph for body
      vars_for_body = switch_vars.map { |x| TensorStream.identity(x[1]) }
      @pivot_for_body = vars_for_body[0]
      vars_for_body_with_tensor_arrays = convert_flows_to_tensorarrays(flat_loop_vars, vars_for_body)
      packed_vars_for_body = pack_sequence_as(original_loop_vars, vars_for_body_with_tensor_arrays)
      body_result = body.call(*packed_vars_for_body)
      body_result = [body_result] unless body_result.is_a?(Array)
      original_body_result = body_result
      result = map_structure(->(x) { convert_tensorarray_to_flow(x) }, _flatten(body_result))
      result = TensorStream.convert_n_to_tensor_or_indexed_slices(result)

      raise TensorStream::ValueError, "Number of inputs and outputs of body must match loop_vars: #{merge_vars.size}, #{result.size}" if merge_vars.size != result.size

      next_vars = []
      merge_vars.zip(result) do |m, v|
        next_vars << add_next_and_back_edge(m, v)
      end

      exit_vars = switch_vars.map { |x| _exit(x[0]) }
      @loop_exits = exit_vars
      exit_result(exit_vars)
      [original_body_result, exit_vars]
    end

    def fix_control_inputs_and_context(enters)
      graph = TensorStream.get_default_graph
      enters.each do |e|
        if e.is_a?(Tensor)
          xs = [e]
        else
          raise TensorStream::TypeError, "Type #{s.class} is not supported"
        end
        xs.each do |x|
          inp_op = x.op.inputs[0].op
          control_inputs = graph._control_dependencies_for_inputs([inp_op])
          outer_control_inputs = control_inputs.select { is_in_outer_context(op) }
          x.op.control_flow_context = self
          x.op.add_control_inputs(outer_control_inputs)
          graph._record_op_seen_by_control_dependencies(x.op)
        end
      end
    end

    def add_next_and_back_edge(m, v, enforce_shape_invariant: true)
      raise TensorStream::TypeError, "Type #{m.class.name} not supported" unless m.is_a?(Tensor)

      v = TensorStream.convert_to_tensor(v)
      v = _next_iteration(v)
      m.op.set_input(1, v)
    end

    def _next_iteration(data, name: nil)
      data = TensorStream.convert_to_tensor(data)
      _op(:next_iteration, data, name: name)
    end

    def set_shape_invariants(input_vars, enter_vars, shapes)
      return if shapes.nil?

      flat_shapes = _flatten(shapes)

      raise TensorStream::ValueError, "shapes must be a (possibley nested) list of shapes." if flat_shapes.detect { |s| !s.is_a?(TensorShape) }

      input_vars.zip(enter_vars, flat_shapes) do |inp, var, shape|
        if var.is_a?(Tensor)
          raise TensorStream::ValueError, "The shape invariant specified for #{inp.name} is not compatible with the initial shape of the loop variable" if !_shape_less_than_or_equal(inp.shape, shape)
          var.set_shape(shape)
        else
          raise TensorStream::TypeError("Type #{var.class} is not supported")
        end
      end
    end

    def convert_tensorarray_to_flow(tensor_or_tensor_array)
      return tensor_or_tensor_array.flow if tensor_or_tensor_array.is_a?(TensorArray)

      tensor_or_tensor_array
    end

    def convert_flows_to_tensorarrays(tensors_or_tensorarrays, tensors_or_flows)
      raise TensorStream::ValueError, "Lengths of original Tensor list and new list do not match: #{tensor_or_tensor_arrays.size} vs. #{tensors_or_flows.size}" if tensors_or_tensorarrays.size != tensors_or_flows.size

      tensors_or_tensorarrays.zip(tensors_or_flows).map do |ta, t_or_flow|
        ta.is_a?(TensorArray) ? _make_tensor_array(ta, t_or_flow) : t_or_flow
      end
    end

    def switch_ref_or_tensor(data, pred, name: 'switch')
      data = TensorStream.convert_to_tensor(data)
      switch(data, pred, name: name)
    end

    def maybe_add_control_dependency(op)
      is_op_free = ->(op) {
        return false if !op.control_inputs.empty?
        op.inputs.each do |x|
          return false if !is_loop_constant_enter(x.op)
        end
        true
      }

      op.control_input << get_control_pivot.op if is_op_free(op)
    end

    def add_forward_loop_counter(outer_grad_state)
      n = TensorStream.conant(0, name: 'f_count')
      if outer_grad_state
        outer_add_op = outer_grad_state.forward_index.op.inputs[0].op
        n.op.control_input << outer_add_op
      end
      enter
      add_name(n.name)
      enter_n = _enter(n, @name, is_contant: false, parallel_iterations: @parallel_iterations, name: 'f_count')
      merge_n = merge([enter_n, enter_n])[0]
      switch_n = switch(merge_n, @pivot)

      index = TensorStream.add(switch_n[1], 1)
      next_n = _next_iteration(index)
      merge_n.op.update_input(1, next_n)
      total_iterations = exit(switch_n[0], name: 'f_count')
      @loop_exits << total_iterations
      exit_result([total_iterations])
      exit
      [total_iterations, next_n]
    end

    ##
    # Add the backprop loop that controls the iterations.
    def add_back_prop_loop_counter(count, outer_grad_state)
      one = TensorStream.constant(1, name: 'b_count')
      enter
      add_name(count.name)
      enter_count = enter(count, @name, is_constant: false, parallel_iterations: @parallel_iterations, name: 'b_count')
      merge_count = merge([enter_count, enter_count])[0]
      @pivot_for_pred = merge_count
      pred = TensorStream.greater_equal(merge_count, one)
      @pivot = loop_count(pred, name: 'b_count')
      switch_count = switch(merge_count, @pivot)

      index = TensorStream.subtract(switch_count[1], one)
      @pivot_for_body = index
      next_count = _next_iteration(index)
      merge_count.op.update_input(1, next_count)
      final_zero = exit(switch_count[0], name: 'b_count')
      @loop_exits << final_zero
      outer_grad_state.grad_sync.control_input << final_zero.op if outer_grad_state

      exit_result([final_zero])
      exit
      next_count
    end

    def add_back_prop_accumulator(op, grad)
      exit
      shape = grad.shape
      if shape.fully_defined?
        @outer_context.enter if @outer_context
        acc = TensorStream.constant(0, grad.data_type, shape: shape, name: 'b_acc')
        @outer_context.exit if @outer_context
      else
        value = op.inputs[0]
        if @outer_context.is_a?(WhileContext) && @outer_context.grad_state
          forward_ctxt = @grad_state.forward_context
          forward_ctxt.outer_context.enter
          zero_shape = TensorStream.shape_internal(value, optimize: false)
          forward_ctxt.outer_context.exit
          outer_grad_state = @grad_state.outer_grad_state
          history_zeros_shape = outer_grad_state.add_forward_accumulator(zeros_shape)
          @outer_context.enter
          real_shape = outer_grad_state.add_back_prop_accumulated_value(history_zeros_shape, zero_shape)
          acc = TensorStream.zeros(real_shape, grad.data_type)
          @outer_context.exit
        else
          @outer_context.enter if @outer_context
          zeros_shape = TensorStream.shape_inteernal(value, optimize: false)
          acc = TensorStream.zeros(zeros_shape, grad.data_type)
          @outer_context.exit if @outer_context
        end
        acc.shape = grad.shape
      end

      enter
      add_name(acc.name)
      enter_acc = _enter(acc, @name, is_constant: false, parallel_iterations: @parallel_iterations, name: 'b_acc')
      merge_acc = merge([enter_acc, enter_acc], name: 'b_acc')[0]
      switch_acc_false, swith_acc_true = switch(merge_acc, @pivot)
      add_acc = TensorStram.add(switch_acc_true, grad)
      next_acc = _next_iteration(add_acc)
      merge_acc.op.upate_input(1, next_acc)

      acc_result = exit(switch_acc_false, name: 'b_acc')
      @loop_exits << acc_result
      exit_result([acc_result])
      acc_result
    end

    def initialize_values(values)
      @values = Set.new
      values.each do |x|
        if x.is_a?(Tensor)
          @values << x.name
        else
        end
      end
    end

    def init_from_args(maximum_iterations, parallel_iterations, back_prop, swap_memory, name)
      raise TensorStream::ValueError, "`parallel_iterations` must be a positive integer: #{parallel_iterations}" if !parallel_iterations.is_a?(Integer) || (parallel_iterations <= 0)

      @name = TensorStream.get_default_graph.unique_name(name)
      @parallel_iterations = parallel_iterations
      @maximum_iterations = maximum_iterations
      @back_prop = back_prop
      @swap_memory = swap_memory
      @pivot_for_pred = nil
      @pivot_for_body = nil
      @pivot = nil
      @loop_exits = []
    end
  end
  class ControlFlow < Operation
    attr_accessor :ops

    def initialize(flow_type, inputs, ops = nil, options = {})
      setup_initial_state(options)
      @options = options
      @operation = :"flow_#{flow_type}"
      @inputs = inputs
      @name = [@graph.get_name_scope, options[:name] || set_name].compact.join("/")
      @ops = ops
      @consumers = Set.new
      @shape = TensorShape.new([inputs.size])
      @graph.add_node(self)
    end

    def set_data_type(_passed_data_type)
      :unknown
    end

    def run
      eval
    end
  end
end
