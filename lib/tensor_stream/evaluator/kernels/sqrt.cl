
__kernel void sqrt_fp(const int M, const int N, __global const float *A, __global float *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    C[globalRow * N + globalCol] = sqrt(A[globalRow * N + globalCol]);
}