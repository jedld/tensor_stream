% c_dtype = dtype_to_c_type(dtype)
__kernel void softmax_grad_<%= dtype %>(const int N,
                      const __global <%= c_dtype %>* A,
                      const __global <%= c_dtype %>* G,
                      __global <%= c_dtype %>* C) {

    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)

    // Compute a single element (loop over K)
    float acc = 0.0f;
    float max = FLT_MIN;
    float row[<%= size %>];
    float grads[<%= size %>][<%= size %>];

    for (int k=0; k<N; k++) {
      max = A[globalRow*N + k] > max ? A[globalRow*N + k] : max;
    }

    for (int k=0; k<N; k++) {
      acc += exp(A[globalRow*N + k] - max);
    }

    // Store the result
    for (int k=0; k < N; k++) {
      row[k] = exp(A[globalRow*N + k] - max) / acc;
    }

    for (int a=0; a < N; a++) {
      for(int b=0; b < N; b++) {
        if (a != b) {
          grads[a][b] = -row[a] * row[b];
        } else {
          grads[a][b] = row[a] * (1.0f - row[a]);
        }
      }
    }

    for (int k=0; k < N; k++) {
      float total_grad = 0.0f;
      for (int a = 0; a < N; a++) {
        total_grad += grads[a][k] * G[globalRow*N + a];
      }
      C[globalRow*N + k] = total_grad;
    }
}