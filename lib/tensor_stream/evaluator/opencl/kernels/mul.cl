% c_dtype = dtype_to_c_type(dtype)
% op = operator_to_c('mul')
<%= render 'operand.cl', c_dtype: c_dtype, op: op, fname: 'mul', dtype: "#{a}_#{b}", result_t: c_dtype %>