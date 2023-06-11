#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <immintrin.h>
#include <time.h>


#include "../simple.h"
#include <CL/cl.h>

struct timespec start, stop;

#define REAL double

const REAL a = 0.1;
const REAL b = 0.2;
const REAL c = 0.3;

void printTimeElapsed( char *text)
{
  double elapsed = (stop.tv_sec -start.tv_sec)*1000.0
                  + (double)(stop.tv_nsec -start.tv_nsec)/1000000.0;
  printf( "%s: %f msec\n", text, elapsed);
}

void timeDirectImplementation( int count, float* data, float* results)
{
  clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &start);
  for (int i = 0; i < count; i++)
    results[i] = data[i] * data[i];
  clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &stop);
  printTimeElapsed( "kernel equivalent on host");
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Please specify 2 arguments (n, iterations).\n");
        return EXIT_FAILURE;
    }

    size_t n = atoll(argv[1]);
    int iterations = atoi(argv[2]);

    cl_int err;
    cl_kernel kernel;
    size_t global[1];
    size_t local[1];
    
    char *KernelSource = readOpenCL("src/opencl/stencil.cl");

    float *data = NULL;                /* Original data set given to device.  */
    float *results = NULL;             /* Results returned from device.  */
    
    data = calloc(n, sizeof(float));
    data[0] = 100;
    data[n - 1] = 1000;
    results = calloc(n, sizeof(float));
    results[0] = data[0];
    results[n - 1] = data[n - 1];
   
    int count = atoi(argv[1]);

    // creates context and command queue, chooses device and platform
    err = initGPU();

    if(err == CL_SUCCESS) {
        global[0] = 512;
        local[0] = 256;
        
        cl_kernel kernel;
        
        for (int i = 0; i < iterations; i++) {
            kernel = setupKernel(KernelSource, "stencil", 3, 
            FloatArr, count, data, 
            FloatArr, count-1, results, 
            IntConst, count);
            runKernel(kernel, 1, global, local);

            if (i != iterations) { 
              float *temp = data;
              data = results;
              results = temp;
            } 
        }

        // printf("Contents of results:\n");
        // for (int i = 0; i < n; i ++) {
        //     printf("index %d: %lf \n",i,results[i]);
        // }

        printKernelTime();
        printTransferTimes();
        
        err = clReleaseKernel (kernel);
        err = freeDevice();
  } 
    return EXIT_SUCCESS;
}
