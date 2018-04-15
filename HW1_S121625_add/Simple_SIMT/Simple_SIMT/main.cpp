#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cl.h>
#include <omp.h>

#include "my_OpenCL_util.h"


#define OPENCL_C_PROG_FILE_NAME1 "simple_kernel.cl"
#define KERNEL_NAME1 "reduction1"

#define OPENCL_C_PROG_FILE_NAME2 "simple_kernel2.cl"
#define KERNEL_NAME2 "reduction2"

#define OPENCL_C_PROG_FILE_NAME3 "simple_kernel3.cl"
#define KERNEL_NAME3 "reduction3"

#define OPENCL_C_PROG_FILE_NAME4 "simple_kernel4.cl"
#define KERNEL_NAME4 "reduction4"

#define OPENCL_C_PROG_FILE_NAME5 "simple_kernel5.cl"	
#define KERNEL_NAME5 "reduction5"


#define ATOMIC 0

//////////////////////////////////////////////////////////////////////////
void generate_random_float_array(float *array, int n) {
	int thread_num = omp_get_max_threads();
	int i;
    srand((unsigned int)201803); 
#pragma omp parallel for default(none) num_threads(thread_num) private(i)
	for (i = 0; i < n; i++) {
        array[i] = 2.0*((float)rand() / RAND_MAX-0.5);
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
    float *array_A, *array_sum;
    OPENCL_C_PROG_SRC prog_src[6];
    
    cl_platform_id platform[2];
    cl_device_id dev[2];
    cl_context context[6];
    cl_command_queue cmd_queues[6];
    cl_program program[6];
    cl_kernel kernel[6]; 
	cl_mem buffer_A, buffer_temp, buffer_sum_GPU;
	cl_mem buffer_A_CPU, buffer_sum_CPU;
    cl_event event_for_timing;
	cl_event event_for_timing_CPU;
    
    if (0) {
        show_OpenCL_platform();
        return 0;
    }
	
    n_elements = 1024*1024*16;

	/* GPU 1dim */
#if ATOMIC == 1
	work_group_size_GPU = 1024;
#else 
	work_group_size_GPU = 128;
#endif

	n_groups = n_elements / work_group_size_GPU;
	
	/* CPU*/
	work_group_size_CPU = 64;
	n_groups_CPU = n_elements / work_group_size_CPU;
	
	/* GPU 2dim */
#if ATOMIC == 1
	size_t local[2] = { 32,32 }; 
#else 
	size_t local[2] = { 32,4 };
#endif
	size_t global[2] = { 1024,1024 * 16 };

    array_A = (float *)malloc(sizeof(float)*n_elements);
    array_sum = (float *)malloc(sizeof(float)*n_groups);
	
    fprintf(stdout, "Generating random input arrays with %d elements each...\n", (int) n_elements);
    generate_random_float_array(array_A, (int) n_elements);
	fprintf(stdout, "Done!\n");
    
    /* NO OPENCL */
    fprintf(stdout, "\n==================== NO OPENCL ====================\n");
    fprintf(stdout, "[CPU Execution] \n");

    CHECK_TIME_START;
	reduction_NO_OPENCL(array_A, array_sum, (int) n_elements);
    //reduction_KahanSUM(array_A, array_sum, (int)n_elements);
	CHECK_TIME_END(compute_time);

    fprintf(stdout, " * Time by host clock = %.3fms\n", compute_time);
    fprintf(stdout, "[Check Results] \n");
	fprintf(stdout, "* sum = %f\n\n", array_sum[0]);
	array_sum[0] = 0.0f;

	clGetPlatformIDs(2, platform, NULL);													
	clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 1, &dev[0], NULL);		
    clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_CPU, 1, &dev[1], NULL);
	/* GPU */
    context[0] = clCreateContext(NULL, 1, &dev[0], NULL, NULL, &errcode_ret);
	context[2] = clCreateContext(NULL, 1, &dev[0], NULL, NULL, &errcode_ret);
	context[3] = clCreateContext(NULL, 1, &dev[0], NULL, NULL, &errcode_ret);
	context[4] = clCreateContext(NULL, 1, &dev[0], NULL, NULL, &errcode_ret);
	context[5] = clCreateContext(NULL, 1, &dev[0], NULL, NULL, &errcode_ret);
	/* CPU */
	context[1] = clCreateContext(NULL, 1, &dev[1], NULL, NULL, &errcode_ret);
	
	/* GPU */
	cmd_queues[0] = clCreateCommandQueue(context[0], dev[0], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
	cmd_queues[2] = clCreateCommandQueue(context[2], dev[0], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
	cmd_queues[3] = clCreateCommandQueue(context[3], dev[0], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
	cmd_queues[4] = clCreateCommandQueue(context[4], dev[0], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
	cmd_queues[5] = clCreateCommandQueue(context[5], dev[0], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
	/* CPU */
    cmd_queues[1] = clCreateCommandQueue(context[1], dev[1], CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
    
	/* GPU */
    prog_src[0].length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME1, &prog_src[0].string);
	prog_src[2].length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME2, &prog_src[2].string);
	prog_src[3].length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME3, &prog_src[3].string);
	prog_src[4].length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME4, &prog_src[4].string);
	prog_src[5].length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME5, &prog_src[5].string);
	/* CPU */
	prog_src[1].length = read_kernel_from_file(OPENCL_C_PROG_FILE_NAME1, &prog_src[1].string);
	
	/* GPU */
	program[0] = clCreateProgramWithSource(context[0], 1, (const char **) &prog_src[0].string, &prog_src[0].length, &errcode_ret);
	program[2] = clCreateProgramWithSource(context[2], 1, (const char **)&prog_src[2].string, &prog_src[2].length, &errcode_ret);
	program[3] = clCreateProgramWithSource(context[3], 1, (const char **)&prog_src[3].string, &prog_src[3].length, &errcode_ret);
	program[4] = clCreateProgramWithSource(context[4], 1, (const char **)&prog_src[4].string, &prog_src[4].length, &errcode_ret);
	program[5] = clCreateProgramWithSource(context[5], 1, (const char **)&prog_src[5].string, &prog_src[5].length, &errcode_ret);
	/* CPU */
	program[1] = clCreateProgramWithSource(context[1], 1, (const char **) &prog_src[1].string, &prog_src[1].length, &errcode_ret);
	
	/* GPU */
    clBuildProgram(program[0], 1, &dev[0], NULL, NULL, NULL);
	clBuildProgram(program[2], 1, &dev[0], NULL, NULL, NULL);
	clBuildProgram(program[3], 1, &dev[0], NULL, NULL, NULL);
	clBuildProgram(program[4], 1, &dev[0], NULL, NULL, NULL);
	clBuildProgram(program[5], 1, &dev[0], NULL, NULL, NULL);
	/* CPU */
	clBuildProgram(program[1], 1, &dev[1], NULL, NULL, NULL);

	/* GPU */
    kernel[0] = clCreateKernel(program[0], KERNEL_NAME1, &errcode_ret);
	kernel[2] = clCreateKernel(program[2], KERNEL_NAME2, &errcode_ret);
	kernel[3] = clCreateKernel(program[3], KERNEL_NAME3, &errcode_ret);
	kernel[4] = clCreateKernel(program[4], KERNEL_NAME4, &errcode_ret);
	kernel[5] = clCreateKernel(program[5], KERNEL_NAME5, &errcode_ret);
	/* CPU */
	kernel[1] = clCreateKernel(program[1], KERNEL_NAME1, &errcode_ret);
	

	/* START *///////////////////////////////////////////////////////////////////////////
	buffer_A = clCreateBuffer(context[0], CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	buffer_sum_GPU = clCreateBuffer(context[0], CL_MEM_READ_WRITE, sizeof(float)*n_groups, NULL, &errcode_ret);
	clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &buffer_A);
	clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &buffer_sum_GPU);

	fprintf(stdout, "\n==================== GPU GLOBAL 1-DIM ====================\n");
	clFinish(cmd_queues[0]);
	CHECK_TIME_START;
	clEnqueueWriteBuffer(cmd_queues[0], buffer_A, CL_FALSE, 0, sizeof(float)*n_elements, array_A, 0, NULL, NULL);
    clFinish(cmd_queues[0]);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Data Transfer to GPU] : %.3fms\n\n", compute_time);
	
	clFinish(cmd_queues[0]);
	CHECK_TIME_START;
    errcode_ret = clEnqueueNDRangeKernel(cmd_queues[0], kernel[0], 1, NULL,
		&n_elements, &work_group_size_GPU, 0, NULL, &event_for_timing);
    clFinish(cmd_queues[0]);  
    CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Kernel Execution] : %.3fms\n\n", compute_time);
    print_device_time(event_for_timing);
	
	clFinish(cmd_queues[0]);
	CHECK_TIME_START;
    clEnqueueReadBuffer(cmd_queues[0], buffer_sum_GPU, CL_TRUE, 0, sizeof(float)*n_groups, array_sum, 0, NULL, &event_for_timing);
    CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Data Transfer] : %.3fms\n\n", compute_time);
    print_device_time(event_for_timing);

#if ATOMIC == 0
	CHECK_TIME_START;
	for (int i = 1; i < n_groups; i++)
		array_sum[0] += array_sum[i];
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "* Time by host clock = %.3fms\n\n", compute_time);
#endif
    fprintf(stdout, "[Check Results] %f\n\n", array_sum[0]);
	memset(array_sum, 0, sizeof(float)*n_groups); 

	////////////////////////////////////////////////////////////////////////////
	buffer_A = clCreateBuffer(context[2], CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	buffer_sum_GPU = clCreateBuffer(context[2], CL_MEM_READ_WRITE, sizeof(float)*n_groups, NULL, &errcode_ret);
	clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &buffer_A);
	clSetKernelArg(kernel[2], 1, sizeof(cl_mem), &buffer_sum_GPU);

	fprintf(stdout, "\n==================== GPU LOCAL 1-DIM ====================\n");
	CHECK_TIME_START;
	clEnqueueWriteBuffer(cmd_queues[2], buffer_A, CL_FALSE, 0, sizeof(float)*n_elements, array_A, 0, NULL, NULL);
	clFinish(cmd_queues[2]);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Data Transfer to GPU] : %.3fms\n\n", compute_time);

	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[2], kernel[2], 1, NULL,
		&n_elements, &work_group_size_GPU, 0, NULL, &event_for_timing);
	clFinish(cmd_queues[2]);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Kernel Execution] : %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	CHECK_TIME_START;
	clEnqueueReadBuffer(cmd_queues[2], buffer_sum_GPU, CL_TRUE, 0, sizeof(float)*n_groups, array_sum, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Data Transfer] : %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

#if ATOMIC == 0
	CHECK_TIME_START;
	for (int i = 1; i < n_groups; i++)
		array_sum[0] += array_sum[i];
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "* Time by host clock = %.3fms\n\n", compute_time);
#endif
	fprintf(stdout, "[Check Results] %f\n\n", array_sum[0]);
	memset(array_sum, 0, sizeof(float)*n_groups);

	///////////////////////////////////////////////////////////////////////////////
	buffer_A = clCreateBuffer(context[3], CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	buffer_sum_GPU = clCreateBuffer(context[3], CL_MEM_READ_WRITE, sizeof(float)*n_groups, NULL, &errcode_ret);
	clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &buffer_A);
	clSetKernelArg(kernel[3], 1, sizeof(cl_mem), &buffer_sum_GPU);


	fprintf(stdout, "\n==================== GPU GLOBAL 2-DIM ====================\n");
	CHECK_TIME_START;
	errcode_ret=clEnqueueWriteBuffer(cmd_queues[3], buffer_A, CL_FALSE, 0, sizeof(float)*n_elements, (const void*)array_A, 0, NULL, NULL);
	CHECK_ERROR_CODE(errcode_ret);
	clFinish(cmd_queues[3]);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Data Transfer to GPU] : %.3fms\n\n", compute_time);

	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[3], kernel[3], 2, NULL,
		global, local, 0, NULL, &event_for_timing);
	clFinish(cmd_queues[3]);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Kernel Execution] : %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	CHECK_TIME_START;
	clEnqueueReadBuffer(cmd_queues[3], buffer_sum_GPU, CL_TRUE, 0, sizeof(float)*n_groups, array_sum, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Data Transfer] : %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

#if ATOMIC == 0
	CHECK_TIME_START;
	for (int i = 1; i < n_groups; i++)
		array_sum[0] += array_sum[i];
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "* Time by host clock = %.3fms\n\n", compute_time);
#endif
	fprintf(stdout, "[Check Results] %f\n\n", array_sum[0]);
	memset(array_sum, 0, sizeof(float)*n_groups);
	///////////////////////////////////////////////////////////////////////////////
	buffer_A = clCreateBuffer(context[4], CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	buffer_sum_GPU = clCreateBuffer(context[4], CL_MEM_READ_WRITE, sizeof(float)*n_groups, NULL, &errcode_ret);
	clSetKernelArg(kernel[4], 0, sizeof(cl_mem), &buffer_A);
	clSetKernelArg(kernel[4], 1, sizeof(cl_mem), &buffer_sum_GPU);

	fprintf(stdout, "\n==================== GPU LOCAL 2-DIM ====================\n");
	CHECK_TIME_START;
	errcode_ret = clEnqueueWriteBuffer(cmd_queues[4], buffer_A, CL_FALSE, 0, sizeof(float)*n_elements, (const void*)array_A, 0, NULL, NULL);
	CHECK_ERROR_CODE(errcode_ret);
	clFinish(cmd_queues[4]);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Data Transfer to GPU] : %.3fms\n\n", compute_time);

	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[4], kernel[4], 2, NULL,
		global, local, 0, NULL, &event_for_timing);
	clFinish(cmd_queues[4]);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Kernel Execution] : %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	CHECK_TIME_START;
	clEnqueueReadBuffer(cmd_queues[4], buffer_sum_GPU, CL_TRUE, 0, sizeof(float)*n_groups, array_sum, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Data Transfer] : %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

#if ATOMIC == 0
	CHECK_TIME_START;
	for (int i = 1; i < n_groups; i++)
		array_sum[0] += array_sum[i];
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "* Time by host clock = %.3fms\n\n", compute_time);
#endif
	fprintf(stdout, "[Check Results] %f\n\n", array_sum[0]);
	memset(array_sum, 0, sizeof(float)*n_groups);
	///////////////////////////////////////////////////////////////////////////////
	buffer_A = clCreateBuffer(context[5], CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	buffer_sum_GPU = clCreateBuffer(context[5], CL_MEM_READ_WRITE, sizeof(float)*n_groups, NULL, &errcode_ret);
	clSetKernelArg(kernel[5], 0, sizeof(cl_mem), &buffer_A);
	clSetKernelArg(kernel[5], 1, sizeof(cl_mem), &buffer_sum_GPU);

	fprintf(stdout, "\n==================== GPU SMALL WGS 1-DIM ====================\n");
	CHECK_TIME_START;
	clEnqueueWriteBuffer(cmd_queues[5], buffer_A, CL_FALSE, 0, sizeof(float)*n_elements, array_A, 0, NULL, NULL);
	clFinish(cmd_queues[5]);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Data Transfer to GPU] : %.3fms\n\n", compute_time);

	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[5], kernel[5], 1, NULL,
		&n_elements, &work_group_size_GPU, 0, NULL, &event_for_timing);
	clFinish(cmd_queues[5]);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Kernel Execution] : %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	CHECK_TIME_START;
	clEnqueueReadBuffer(cmd_queues[5], buffer_sum_GPU, CL_TRUE, 0, sizeof(float)*n_groups, array_sum, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Data Transfer] : %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

#if ATOMIC == 0
	CHECK_TIME_START;
	for (int i = 1; i < n_groups; i++)
		array_sum[0] += array_sum[i];
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "* Time by host clock = %.3fms\n\n", compute_time);
#endif
	fprintf(stdout, "[Check Results] %f\n\n", array_sum[0]);
	memset(array_sum, 0, sizeof(float)*n_groups);
	///////////////////////////////////////////////////////////////////////////////
	/*buffer_A_CPU = clCreateBuffer(context[1], CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	buffer_sum_CPU = clCreateBuffer(context[1], CL_MEM_WRITE_ONLY, sizeof(float)*n_groups_CPU, NULL, &errcode_ret);
	clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &buffer_A_CPU);
	clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &buffer_sum_CPU);

    fprintf(stdout, "==================== CPU ====================\n");
	CHECK_TIME_START;
	errcode_ret = clEnqueueWriteBuffer(cmd_queues[1], buffer_A_CPU, CL_FALSE, 0, sizeof(float)*n_elements, array_A, 0, NULL, NULL);
	clFinish(cmd_queues[1]);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Data Transfer] :%.3fms\n\n", compute_time);

    CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues[1], kernel[1], 1, NULL,
		&n_elements, &work_group_size_CPU, 0, NULL, &event_for_timing_CPU);
    clFinish(cmd_queues[1]);
    CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Kernel Execution] : %.3fms\n\n", compute_time);
    print_device_time(event_for_timing_CPU);

    
	CHECK_TIME_START;
    errcode_ret = clEnqueueReadBuffer(cmd_queues[1], buffer_sum_CPU, CL_TRUE, 0,sizeof(float)*n_groups_CPU, array_sum, 0, NULL, &event_for_timing_CPU);
	CHECK_TIME_END(compute_time);
	fprintf(stdout, "[Data Transfer] :%.3fms\n\n", compute_time);
    
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
    fprintf(stdout, "* sum = %f\n\n", array_sum[0]);*/
    //////////////////////////////////////////////////////////////////////////////////

    /* Free OpenCL resources. */
    clReleaseMemObject(buffer_A);
	clReleaseMemObject(buffer_sum_GPU);
	
	clReleaseKernel(kernel[0]);
	clReleaseKernel(kernel[2]);
	clReleaseKernel(kernel[3]);
	clReleaseKernel(kernel[4]);
	clReleaseKernel(kernel[5]);

    clReleaseProgram(program[0]);
	clReleaseProgram(program[2]);
	clReleaseProgram(program[3]);
	clReleaseProgram(program[4]);
	clReleaseProgram(program[5]);

	clReleaseCommandQueue(cmd_queues[0]);
	clReleaseCommandQueue(cmd_queues[2]);
	clReleaseCommandQueue(cmd_queues[3]);
	clReleaseCommandQueue(cmd_queues[4]);
	clReleaseCommandQueue(cmd_queues[5]);

    clReleaseContext(context[0]);
	clReleaseContext(context[2]);
	clReleaseContext(context[3]);
	clReleaseContext(context[4]);
	clReleaseContext(context[5]);

    /* Free host resources. */
    free(array_A);
    free(array_sum);
    free(prog_src[0].string);
	free(prog_src[2].string);
	free(prog_src[3].string);
	free(prog_src[4].string);
	free(prog_src[5].string);
    
/*
	clReleaseMemObject(buffer_A_CPU);
	clReleaseMemObject(buffer_sum_CPU);
	clReleaseKernel(kernel[1]);
	clReleaseProgram(program[1]);
	clReleaseCommandQueue(cmd_queues[1]);
	clReleaseContext(context[1]);
	free(prog_src[1].string);
*/
    return 0;
}



