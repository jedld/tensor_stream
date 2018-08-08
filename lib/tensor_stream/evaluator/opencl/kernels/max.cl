 // same dimension add floating point op
% c_dtype = dtype_to_c_type(dtype)
 __kernel void max_<%= dtype %>_<%= dtype %>(const int M, const int N, const int switch_op, __global const <%= c_dtype %> *A, __global const <%= c_dtype %> *B, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    C[globalRow * N + globalCol] = A[globalRow * N + globalCol] >= B[globalRow * N + globalCol] ? A[globalRow * N + globalCol] : B[globalRow * N + globalCol];
}

 // 1D + Scalar floating point add op
 __kernel void max_c_<%= dtype %>_<%= dtype %>(const int M, const int N, const int switch_op, __global const <%= c_dtype %> *A, __global const <%= c_dtype %> *B, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    
    if (switch_op == 0) {
      C[globalRow * N + globalCol] = A[globalRow * N + globalCol] >= B[0] ? A[globalRow * N + globalCol] : B[0];
    } else {
      C[globalRow * N + globalCol] = B[0] >= A[globalRow * N + globalCol] ? B[0] : A[globalRow * N + globalCol];
    }
}

 // 1D + Scalar floating point add op broadcast
 __kernel void max_b_<%= dtype %>_<%= dtype %>(const int M, const int N, const int M2, const int N2, const int switch_op, __global const <%= c_dtype %> *A, __global const <%= c_dtype %> *B, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    
    int b_m_index = globalRow;
    int b_n_index = globalCol;

    if ( b_m_index >= M2) {
      b_m_index = b_m_index % M2;
    };

    if (b_n_index >= N2) {
      b_n_index = b_n_index % N2;
    }

    if (switch_op == 0) {
      C[globalRow * N + globalCol] = A[globalRow * N + globalCol] >= B[b_m_index * N2 + b_n_index] ? A[globalRow * N + globalCol] : B[b_m_index * N2 + b_n_index];
    } else {
      C[globalRow * N + globalCol] = B[b_m_index * N2 + b_n_index] >= A[globalRow * N + globalCol] ? B[b_m_index * N2 + b_n_index] :  A[globalRow * N + globalCol];
    }
}