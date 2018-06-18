% c_dtype = dtype_to_c_type(dtype)

__kernel void sign_<%= dtype %>(const int M, const int N, __global const <%= c_dtype %> *A, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    <%= c_dtype %> value = A[globalRow * N + globalCol];
% if floating_point?(dtype)
    if (isnan(value) || value == 0.0f) {
      C[globalRow * N + globalCol] = 0.0;
    } else {
      C[globalRow * N + globalCol] = value < 0 ? -1.0 : 1.0;
    }
% else
  if (value == 0) {
    C[globalRow * N + globalCol] = 0;
  } else {
    C[globalRow * N + globalCol] = value < 0 ? -1 : 1;
  }
% end
}