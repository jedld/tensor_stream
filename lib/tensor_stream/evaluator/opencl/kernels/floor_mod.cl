% c_dtype = dtype_to_c_type(dtype)
% op = operator_to_c('mod')
<%= render 'operand.cl', c_dtype: c_dtype, op: op, fname: 'floor_mod', dtype: "#{a}_#{b}", result_t: c_dtype %>