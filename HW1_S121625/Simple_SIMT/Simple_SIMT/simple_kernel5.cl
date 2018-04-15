
__kernel
void reduction5(__global float* A, __global float* sum) {
	int tid = get_global_id(0);
	int lid = get_local_id(0);
	int gid = get_group_id(0);
	int group_size = get_local_size(0);
	
	__local float Asub[256];
	
	Asub[lid] = A[group_size*gid+lid];
	
	for (unsigned int i = group_size / 2; i > 0; i >>= 1) {
		if (lid < i) {
			Asub[lid] += Asub[lid + i];
		}
	}
	if (lid == 0)
		sum[gid] = Asub[lid];
}
