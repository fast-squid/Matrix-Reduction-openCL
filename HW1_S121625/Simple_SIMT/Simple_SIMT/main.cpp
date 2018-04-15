#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cl.h>
#include <omp.h>

#include "my_OpenCL_util.h"

#define KERNEL 4

#if KERNEL == 1
#define OPENCL_C_PROG_FILE_NAME "simple_kernel.cl"
#define KERNEL_NAME "reduction1"
#elif KERNEL == 2
#define OPENCL_C_PROG_FILE_NAME "simple_kernel2.cl"
#define KERNEL_NAME "reduction2"
#elif KERNEL == 3
#define OPENCL_C_PROG_FILE_NAME "simple_kernel3.cl"
#define KERNEL_NAME "reduction3"
#elif KERNEL == 4
#define OPENCL_C_PROG_FILE_NAME "simple_kernel4.cl"
#define KERNEL_NAME "reduction4"
#elif KERNEL == 5									/* NO SYNCH */
#define OPENCL_C_PROG_FILE_NAME "simple_kernel5.cl"	
#define KERNEL_NAME "reduction5"
#endif

#define ATOMIC 0

//////////////////////////////////////////////////////////////////////////
void generate_random_float_array(float *array, int n) {
	int thread_num = omp_get_max_threads();
	int i;

    srand((unsigned int)201803); 
#pragma omp parallel for default(none) num_threads(thread_num) private(i)
	for (i = 0; i < n; i++) {
        array[i] = 10*((float)rand() / RAND_MAX-0.5);
    }
}

void reduction_NO_OPENCL(float *A,float* array_sum, int n) {
	array_sum[0] = 0.0f;
	for (int i = 0; i < n; i++) {
		array_sum[0] += A[i];
    }
}

void reduction_KahanSUM(float *A, float* array_sum, int n) {
	float c = 0.0f, temp1, temp2;
	array_sum[0] = 0.0f;
	for (int i = 0; i < n; i++) {
		temp2 = A[i] - c; 
		temp1 = array_sum[0] + temp2;
		c = temp1 - array_sum[0] - temp2;
		array_sum[0] = temp1;
	}
}
//////////////////////////////////////////////////////////////////////////

#define IDX_GPU 0
#define IDX_CPU 1

typedef struct _OPENCL_C_PROG_SRC {
    size_t length;
    char *string;
} OPENCL_C_PROG_SRC;

int main(void) {
    cl_int errcode_ret;
    float compute_time;
	size_t n_elements, work_group_size_GPU, work_group_size_CPU;
	size_t n_groups,n_groups_CPU;
    float *array_A, *array_temp, *array_sum;
    OPENCL_C_PROG_SRC prog_src;
    
    cl_platform_id platform[2];
    cl_device_id dev[2];
    cl_context context[2];
    cl_command_queue cmd_queues[2];
    cl_program program[2];
    cl_kernel kernel[2];
	cl_mem buffer_A, buffer_temp, buffer_sum_GPU;
	cl_mem buffer_A_CPU, buffer_sum_CPU;
    cl_event event_for_timing;
	cl_event event_for_timing_CPU;
    
    if (0) {
        show_OpenCL_platform();
        return 0;
    }
    
    n_elements = 1024*1024*16;
    work_group_size_GPU = 128;
#if KERNEL == 5
	work_group_size_GPU = 32;
#endif
    work_group_size_CPU = 16; 

	n_groups = n_elements / work_group_size_GPU;
	n_groups_CPU = n_elements / work_group_size_CPU;

    array_A = (float *)malloc(sizeof(float)*n_elements);
    array_sum = (float *)malloc(sizeof(float)*n_elements);
	
    fprintf(stdout, "Generating random input arrays with %d elements each...\n", (int) n_elements);
    generate_random_float_array(array_A, (int) n_elements);
    fprintf(stdout, "Done!\n");
    
    /* NO OPENCL */
    fprintf(stdout, "\n==================== NO OPENCL ====================\n");
    fprintf(stdout, "[CPU Execution] \n");

    CHECK_TIME_START;
	//reduction_NO_OPENCL(array_A, array_sum, (int) n_elements);
    reduction_KahanSUM(array_A, array_sum, (int)n_elements);
	
	CHECK_TIME_END(compute_time);

    fprintf(stdout, " * Time by host clock = %.3fms\n", compute_time);
    fprintf(stdout, "[Check Results] \n");
	fprintf(stdout, "* sum = %f\n\n", array_sum[0]);
	array_sum[0] = 0.0f;

	clGetPlatformIDs(2, platform, NULL);													
	clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 1, &dev[0], NULL);		
    clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_CPU, 1, &dev[1], NULL);
    
    context[0] = clCreateContext(NULL, 1, &dev[0], NULL, NULL, &errcode_ret);
	context[1] = clCreateContext(NULL, 1, &dev[1], NULL, NULL, &errcode_ret);
    
	cmd_queues[0] = clCreateCommandQueue(context[0], dev[0], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
    cmd_queues[1] = clCreateCommandQueue(context[1], dev[1], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
    
    prog_src.length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME, &prog_src.string);
    program[0] = clCreateProgramWithSource(context[0], 1, (const char **) &prog_src.string, &prog_src.length, &errcode_ret);
	program[1] = clCreateProgramWithSource(context[1], 1, (const char **) &prog_src.string, &prog_src.length, &errcode_ret);
    
    clBuildProgram(program[0], 1, &dev[0], NULL, NULL, NULL);
	clBuildProgram(program[1], 1, &dev[1], NULL, NULL, NULL);
    
    kernel[0] = clCreateKernel(program[0], KERNEL_NAME, &errcode_ret);
	kernel[1] = clCreateKernel(program[1], KERNEL_NAME, &errcode_ret);
    
	buffer_A = clCreateBuffer(context[0], CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
    buffer_sum_GPU = clCreateBuffer(context[0], CL_MEM_READ_WRITE, sizeof(float)*n_groups, NULL, &errcode_ret);
	
	buffer_A_CPU = clCreateBuffer(context[1], CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	buffer_sum_CPU = clCreateBuffer(context[1], CL_MEM_WRITE_ONLY, sizeof(float)*n_groups_CPU, NULL, &errcode_ret);

	clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &buffer_A);
	clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &buffer_sum_GPU);
	clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &buffer_A_CPU);
	clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &buffer_sum_CPU);

	fprintf(stdout, "\n==================== GPU ====================\n");
	fprintf(stdout, "[Data Transfer to GPU] \n");
	
	CHECK_TIME_START;
	clEnqueueWriteBuffer(cmd_queues[IDX_GPU], buffer_A, CL_FALSE, 0, sizeof(float)*n_elements, array_A, 0, NULL, NULL);
    clFinish(cmd_queues[IDX_GPU]);
	CHECK_TIME_END(compute_time);
	
	fprintf(stdout, "* Time by host clock = %.3fms\n\n", compute_time);
	fprintf(stdout, "[Kernel Execution] \n");
    
	
    CHECK_TIME_START;
#if KERNEL == 3
	size_t local[2] = { 32,4}; // 32, 4 : 5.187
	size_t global[2] = { 1024,1024*16};
	n_groups = n_elements / (32*4);
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[IDX_GPU], kernel[0], 2, NULL,
		global, local, 0, NULL, &event_for_timing);
#elif KERNEL == 4
	size_t local[2] = { 32, 4 };
	size_t global[2] = { 1024,1024 * 16 };
	n_groups = n_elements / (4*32);
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[IDX_GPU], kernel[0], 2, NULL,
		global, local, 0, NULL, &event_for_timing);
#else
    errcode_ret = clEnqueueNDRangeKernel(cmd_queues[0], kernel[0], 1, NULL,
		&n_elements, &work_group_size_GPU, 0, NULL, &event_for_timing);
#endif

    CHECK_ERROR_CODE(errcode_ret);
    clFinish(cmd_queues[0]);  
    CHECK_TIME_END(compute_time);

    fprintf(stdout, "* Time by host clock = %.3fms\n\n", compute_time);
    print_device_time(event_for_timing);
    fprintf(stdout, "[Data Transfer] \n");
    
	CHECK_TIME_START;
    clEnqueueReadBuffer(cmd_queues[0], buffer_sum_GPU, CL_TRUE, 0, sizeof(float)*n_groups, array_sum, 0, NULL, &event_for_timing);
    CHECK_TIME_END(compute_time);
    
	fprintf(stdout, "* Time by host clock = %.3fms\n\n", compute_time);
    print_device_time(event_for_timing);
#if ATOMIC == 0
	CHECK_TIME_START;
	for (int i = 1; i < n_groups; i++)
		array_sum[0] += array_sum[i];
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "* Time by host clock = %.3fms\n\n", compute_time);
#endif
    fprintf(stdout, "[Check Results] \n");
    fprintf(stdout, "* sum = %f\n\n", array_sum[0]);
    
	memset(array_sum, 0, sizeof(float)*n_groups); 


    fprintf(stdout, "==================== CPU ====================\n");
	CHECK_TIME_START;
	fprintf(stdout, "[Data Transfer] \n");
	errcode_ret = clEnqueueWriteBuffer(cmd_queues[1], buffer_A_CPU, CL_FALSE, 0, sizeof(float)*n_elements, array_A, 0, NULL, NULL);
	clFinish(cmd_queues[1]);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "* Time by host clock = %.3fms\n\n", compute_time);

    fprintf(stdout, "[Kernel Execution] \n");
    CHECK_TIME_START;
#if KERNEL == 3
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[1], kernel[1], 2, NULL,
		global, local, 0, NULL, &event_for_timing_CPU);
#elif KERNEL == 4
	n_groups_CPU = (1024 * 1024 * 16) / 256;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[1], kernel[1], 2, NULL,
		global, local, 0, NULL, &event_for_timing_CPU);
#else
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[1], kernel[1], 1, NULL,
		&n_elements, &work_group_size_CPU, 0, NULL, &event_for_timing_CPU);
#endif
    clFinish(cmd_queues[1]);
    CHECK_TIME_END(compute_time);

    fprintf(stdout, "* Time by host clock = %.3fms\n", compute_time);
    print_device_time(event_for_timing_CPU);

    fprintf(stdout, "[Data Transfer] \n");  
	CHECK_TIME_START;
    errcode_ret = clEnqueueReadBuffer(cmd_queues[1], buffer_sum_CPU, CL_TRUE, 0,sizeof(float)*n_groups_CPU, array_sum, 0, NULL, &event_for_timing_CPU);
	CHECK_TIME_END(compute_time);
    CHECK_ERROR_CODE(errcode_ret);
    
    fprintf(stdout, "* Time by host clock = %.3fms\n", compute_time);
    print_device_time(event_for_timing_CPU);
#if ATOMIC == 0
	CHECK_TIME_START;
	for (int i = 1; i < n_groups_CPU; i++)
		array_sum[0] += array_sum[i];
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "* Time by host clock = %.3fms\n\n", compute_time);
#endif
    fprintf(stdout, "[Check Results] \n");
    fprintf(stdout, "* sum = %f\n\n", array_sum[0]);
    
    /* Free OpenCL resources. */
    clReleaseMemObject(buffer_A);
	clReleaseMemObject(buffer_A_CPU);
    clReleaseMemObject(buffer_sum_GPU);
    clReleaseMemObject(buffer_sum_CPU);
    clReleaseKernel(kernel[0]);
    clReleaseProgram(program[0]);
	clReleaseKernel(kernel[1]);
	clReleaseProgram(program[1]);
	clReleaseCommandQueue(cmd_queues[IDX_GPU]);
    clReleaseCommandQueue(cmd_queues[IDX_CPU]);
    clReleaseContext(context[0]);
	clReleaseContext(context[1]);
    
    /* Free host resources. */
    free(array_A);
    free(array_sum);
    free(prog_src.string);
    
    return 0;
}



