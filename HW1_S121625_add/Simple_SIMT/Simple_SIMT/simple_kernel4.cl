#define ATOMIC 0
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
void reduction4(__global float* A, __global float* sum) {
	int row = get_local_id(1);
	int col = get_local_id(0);
	const int globalRow = get_global_id(1);
	const int globalCol = get_global_id(0);
	int row_size = get_local_size(1);
	int col_size = get_local_size(0);
	int grow = get_group_id(1);
	int gcol = get_group_id(0);
#if ATOMIC ==0
	__local float Asub[8][64];
#else 
	__local float Asub[32][32];
#endif
	Asub[row][col] = A[globalRow * 1024 + globalCol];
	
	for (unsigned int i = col_size / 2; i > 0; i >>= 1) {
		if (col < i) {
			Asub[row][col] += Asub[row][col+i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	for (unsigned int i = row_size / 2; i > 0; i >>= 1) {
		if (row < i) {
			Asub[row][col] += Asub[row + i][col];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
#if ATOMIC == 0
	if (row == 0 & col == 0)
		sum[grow * col_size + gcol] = Asub[0][0];
#else
	if (row == 0 && col == 0) {
		atomic_add_global(&sum[0], Asub[0][0]);
	}
#endif
}
