% source_ctype = dtype_to_c_type(source_dt)
% target_ctype = dtype_to_c_type(target_dt)

__kernel void cast(const int M, const int N, __global const <%= source_ctype %> *A, __global <%= target_ctype %> *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    C[globalRow * N + globalCol] = A[globalRow * N + globalCol];
}