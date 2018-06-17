// First naive implementation
% c_dtype = dtype_to_c_type(dtype)
__kernel void gemm_<%= dtype %>(const int M, const int N, const int K,
                      const int A_transpose,
                      const int B_transpose,
                      const __global <%= c_dtype %>* A,
                      const __global <%= c_dtype %>* B,
                      __global <%= c_dtype %>* C) {

    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
    <%= c_dtype %> acc = 0.0f;
    for (int k=0; k<K; k++) {
        int a_index = globalRow*K + k;
        int b_index = k*N + globalCol;

        if (A_transpose) {
            a_index = M*k + globalRow;
        }

        if (B_transpose) {
            b_index = globalCol*K + k;
        }
        acc += A[a_index] * B[b_index];
    }

    // Store the result
    C[globalRow*N + globalCol] = acc;
}