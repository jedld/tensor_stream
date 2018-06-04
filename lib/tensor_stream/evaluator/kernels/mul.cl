% %w[fp int].product(%w[mul]).each do |dtype, fname|
% c_dtype = dtype_to_c_type(dtype)
% op = operator_to_c(fname)
<%= render 'operand.cl', c_dtype: c_dtype, op: op, fname: fname, dtype: dtype, result_t: c_dtype %>
% end