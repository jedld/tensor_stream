__kernel void sign_fp(const int M, const int N, __global const float *A, __global float *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    float value = A[globalRow * N + globalCol];
    if (isnan(value) || value == 0.0f) {
      C[globalRow * N + globalCol] = 0.0;
    } else {
      C[globalRow * N + globalCol] = value < 0 ? -1.0 : 1.0;
    }
}

__kernel void sign_int(const int M, const int N, __global const int *A, __global int *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    float value = A[globalRow * N + globalCol];
    if (isnan(value) || value == 0) {
      C[globalRow * N + globalCol] = 0;
    } else {
      C[globalRow * N + globalCol] = value < 0 ? -1 : 1;
    }
}