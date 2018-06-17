% c_dtype = dtype_to_c_type(dtype)
% if TensorStream::Ops::FLOATING_POINT_TYPES.include?(dtype)
__kernel void abs_<%= dtype%>(const int M, const int N, __global const <%= c_dtype %> *A, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    C[globalRow * N + globalCol] = fabs(A[globalRow * N + globalCol]);
}
% else
% %w[int int32].each do |dt|
__kernel void abs_<%= dt %>(const int M, const int N, __global const <%= c_dtype %> *A, __global <%= c_dtype %> *C) {
    // Get the index of the current element to be processed
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    C[globalRow * N + globalCol] = fabs((float)A[globalRow * N + globalCol]);
}
% end
%end