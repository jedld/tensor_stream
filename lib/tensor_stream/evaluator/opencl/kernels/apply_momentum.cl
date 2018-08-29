% c_dtype = dtype_to_c_type(dtype)
 // same dimension add floating point op
 __kernel void apply_momentum_<%= dtype %>(const int M, const int N, __global const <%= c_dtype %> *grad, __global const <%= c_dtype %> *learning_rate,
                                          __global const <%= c_dtype %> *momentum, __global <%= c_dtype %> *output, __global <%= c_dtype %> *acc) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    const int index = globalRow * N + globalCol;
    <%= c_dtype %> acc_m = acc[index];
    acc[index] = acc_m * momentum[0] + grad[index];
<% if nesterov %>
    output[index] -= grad[index] * learning_rate[0] + acc_m * momentum[0] * learning_rate[0];
<% else %>
    output[index] -= acc_m * learning_rate[0];
<% end %>
}