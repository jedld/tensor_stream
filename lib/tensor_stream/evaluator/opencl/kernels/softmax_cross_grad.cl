// First naive implementation
% c_dtype = dtype_to_c_type(dtype)
__kernel void softmax_cross_grad_<%= dtype %>(const int N,
                      const __global <%= c_dtype %>* A,
                      const __global <%= c_dtype %>* L,
                      const __global <%= c_dtype %>* G,
                      __global <%= c_dtype %>* C) {

    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)

    // Compute a single element (loop over K)
    <%= c_dtype %> acc = 0.0f;
    <%= c_dtype %> max = <%= min_value_for(dtype) %>;

    for (int k=0; k<N; k++) {
      max = A[globalRow*N + k] > max ? A[globalRow*N + k] : max;
    }

    for (int k=0; k<N; k++) {
      acc += exp(A[globalRow*N + k] - max);
    }

    // Store the result
    for (int k=0; k < N; k++) {
      C[globalRow*N + k] = ((exp(A[globalRow*N + k] - max)/acc)  *  G[globalRow*N + k] - L[globalRow*N + k]);
    }
}