% ctype = dtype_to_c_type(data_type)

__kernel void pack(const int N, const int index, __global const <%= ctype %> *A, __global <%= ctype %> *C) {
    // Get the index of the current element to be processed
    const int globalCol = get_global_id(0); // Col ID of C (0..N)

    int start = index * <%= divisors[0] %>;
    int ptr = start + globalCol;
    int index_map[<%= divisors.size %>] = { <%= Array.new(divisors.size) { 0 }.join(', ') %> };

    // compute effective coordinates
<% divisors.each_with_index do |div, index| %>
    index_map[<%= index %>] = (int)floor(ptr / (float)<%= div %>);<% if index < divisors.size - 1%>ptr = ptr % <%= div %>;<% end %><% end %>

    // Apply axis translation if needed
<% if axis > 0 %>
    int first = index_map[0];
<% axis.times do |i| %>
    index_map[<%= i %>] = index_map[<%= (i + 1) %>];<% end %>
    index_map[<%= axis %>] = first;
<% end%>

    C[<%= multipliers.each_with_index.map { |m, idx| "#{m}*index_map[#{idx}]" }.join(' + ') %>] = A[globalCol];
}