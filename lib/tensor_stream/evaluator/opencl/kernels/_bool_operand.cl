 // same dimension add floating point op
 __kernel void <%= fname%>_<%= dtype %>(const int M, const int N, const int switch_op, __global const <%= a_dtype %> *A, __global <%= b_dtype %> *B, __global <%= result_t %> *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    C[globalRow * N + globalCol] = A[globalRow * N + globalCol] <%= op %> B[globalRow * N + globalCol] ? 1 : 0;
}

 // 1D + Scalar floating point add op
 __kernel void <%=fname%>_c_<%= dtype %>(const int M, const int N, const int switch_op, __global const <%= a_dtype %> *A, __global <%= b_dtype %> *B, __global <%= result_t %> *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    if (switch_op == 0) {
      C[globalRow * N + globalCol] = A[globalRow * N + globalCol] <%= op %> B[0] ? 1 : 0;
    } else {
      C[globalRow * N + globalCol] = B[0] <%= op %> A[globalRow * N + globalCol] ? 1 : 0;
    }
}

 // 1D + Scalar floating point add op broadcast
 __kernel void <%= fname%>_b_<%= dtype %>(const int M, const int N, const int M2, const int N2, const int switch_op,__global const <%= a_dtype %> *A, __global <%= b_dtype %> *B, __global <%= result_t %> *C) {
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
      C[globalRow * N + globalCol] = A[globalRow * N + globalCol] <%= op %> B[b_m_index * N2 + b_n_index] ? 1 : 0;
    } else {
      C[globalRow * N + globalCol] = B[b_m_index * N2 + b_n_index] <%= op %> A[globalRow * N + globalCol] ? 1 : 0;
    }
}