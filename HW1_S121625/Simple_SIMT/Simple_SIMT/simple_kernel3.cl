void atomic_add_global(volatile global float *source, const float operand) {
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;

	do {
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel
void reduction3(__global float* A, __global float* sum) {
	int row = get_local_id(1);
	int col = get_local_id(0);
	const int globalRow = get_global_id(1);
	const int globalCol = get_global_id(0);
	int row_size = get_local_size(1);
	int col_size = get_local_size(0);
	int grow = get_group_id(1);
	int gcol = get_group_id(0);
	
	int mem_ref = globalRow * 1024;
	for (unsigned int i = col_size / 2; i > 0; i >>= 1) {
		if (col < i) {
			A[mem_ref + globalCol] += A[mem_ref + globalCol + i];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	for (unsigned int i = row_size / 2; i > 0; i >>= 1) {
		if (row < i) {
			A[mem_ref + globalCol] += A[(globalRow + i) * 1024 + globalCol];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	 if (row == 0 && col == 0)
		sum[grow*col_size + gcol] = A[mem_ref + globalCol];
	/*if (row == 0 && col==0) {
		atomic_add_global(&sum[0], A[globalRow*1024+globalCol]);
	}*/
}
