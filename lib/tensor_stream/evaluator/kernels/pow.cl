 // same dimension add floating point op
 __kernel void pow_fp(const int M, const int N, const int switch_op, __global const float *A, __global const float *B, __global float *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    C[globalRow * N + globalCol] = pow((float)A[globalRow * N + globalCol], (float)B[globalRow * N + globalCol]);
}

 // 1D + Scalar floating point add op
 __kernel void pow_c_fp(const int M, const int N, const int switch_op, __global const float *A, __global const float *B, __global float *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    
    if (switch_op == 0) {
      C[globalRow * N + globalCol] = pow((float)A[globalRow * N + globalCol], (float)B[0]);
    } else {
      C[globalRow * N + globalCol] = pow((float)B[0], (float)A[globalRow * N + globalCol]);
    }
}

 // 1D + Scalar floating point add op broadcast
 __kernel void pow_b_fp(const int M, const int N, const int M2, const int N2, const int switch_op, __global const float *A, __global const float *B, __global float *C) {
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
      C[globalRow * N + globalCol] = pow((float)A[globalRow * N + globalCol], (float)B[b_m_index * N2 + b_n_index]);
    } else {
      C[globalRow * N + globalCol] = pow((float)B[b_m_index * N2 + b_n_index], (float)A[globalRow * N + globalCol]);
    }
}

 // same dimension add floating point op
 __kernel void pow_int(const int M, const int N, const int switch_op, __global const int *A, __global const int *B, __global int *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    int acc = A[globalRow * N + globalCol];
    const int count = B[globalRow * N + globalCol];
    const int c = A[globalRow * N + globalCol];

    if (count < 4) {
      for(int i = 0; i < count - 1; i++) {
        acc *= c;
      }
      C[globalRow * N + globalCol] = acc;
    } else {
      C[globalRow * N + globalCol] = pow((float)c, (float)count);
    }
}

 // 1D + Scalar floating point add op
 __kernel void pow_c_int(const int M, const int N, const int switch_op, __global const int *A, __global const int *B, __global int *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
    
    int acc, count, c;
    if (switch_op == 0) {
      acc = A[globalRow * N + globalCol];
      count = B[0];
      c = A[globalRow * N + globalCol];
    } else {
      acc = B[0];
      count = A[globalRow * N + globalCol];
      c = B[0];
    }
    if (count < 4) {
      for(int i =0; i < count - 1; i++) {
        acc *= c;
      }
      C[globalRow * N + globalCol] = acc;
    } else {
      C[globalRow * N + globalCol] = pow((float)c, (float)count);
    }
}

 // 1D + Scalar floating point add op broadcast
 __kernel void pow_b_int(const int M, const int N, const int M2, const int N2, const int switch_op, __global const int *A, __global const int *B, __global int *C) {
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

    int acc, count, c;

    if (switch_op == 0) {
      acc = A[globalRow * N + globalCol];
      count = B[b_m_index * N2 + b_n_index];
      c = A[globalRow * N + globalCol];
    } else {
      acc = B[b_m_index * N2 + b_n_index];
      count = A[globalRow * N + globalCol];
      c = B[b_m_index * N2 + b_n_index];
    }

    if (count < 4) {
      for (int i = 0; i < count - 1; i++) {
        acc *= c;
      }
      C[globalRow * N + globalCol] = acc;
    } else {
      C[globalRow * N + globalCol] = pow((float)c, (float)count);
    }
}