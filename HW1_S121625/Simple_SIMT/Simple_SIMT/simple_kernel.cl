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
void reduction1(__global float* A, __global float* sum) {
	int tid = get_global_id(0);
	int lid = get_local_id(0);
	int gid = get_group_id(0);
	int group_size = get_local_size(0);

	for (unsigned int i = group_size / 2; i > 0; i >>= 1) {
		if (lid < i) {
			A[tid] += A[tid + i];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	if (lid == 0)
		sum[gid] = A[tid];
	/*if (lid == 0) {
		atomic_add_global(&sum[0], A[tid]);
	}*/
	
}
