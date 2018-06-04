 __kernel void argmin_fp(const int M, const int N, const int switch_op, __global const float *A, __global const float *B, __global float *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    
    C[globalRow * N + globalCol] = A[globalRow * N + globalCol] + B[globalRow * N + globalCol];
}

 __kernel void argmin_int(const int M, const int N, const int switch_op, __global const int *A, __global const int *B, __global int *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    
    C[globalRow * N + globalCol] = A[globalRow * N + globalCol] + B[globalRow * N + globalCol];
}