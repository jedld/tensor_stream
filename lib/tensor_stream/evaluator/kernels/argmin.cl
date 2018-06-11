% c_dtype = dtype_to_c_type(dtype)
 __kernel void argmin_<%= dtype %>(const int M, const int N, const int switch_op, __global const <%= c_dtype %> *A, __global const <%= c_dtype %> *B, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    
    C[globalRow * N + globalCol] = A[globalRow * N + globalCol] + B[globalRow * N + globalCol];
}