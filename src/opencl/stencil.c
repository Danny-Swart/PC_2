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

// void Stencil(REAL **in, REAL **out, size_t n, int iterations)
// {

//     (*out)[0] = (*in)[0];
//     (*out)[n - 1] = (*in)[n - 1];

//     for (int t = 1; t <= iterations; t++) {
//         /* Update only the inner values. */
//         for (int i = 1; i < n - 1; i++) {
//             (*out)[i] = a * (*in)[i - 1] +
//                         b * (*in)[i] +
//                         c * (*in)[i + 1];
//         }

//         /* The output of this iteration is the input of the next iteration (if there is one). */
//         if (t != iterations) {
//             REAL *temp = *in;
//             *in = *out;
//             *out = temp;
//         }
//     }
// }

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

    printf("START MAIN");

    size_t n = atoll(argv[1]);
    int iterations = atoi(argv[2]);

    // REAL *in = calloc(n, sizeof(REAL));
    // in[0] = 100;
    // in[n - 1] = 1000;
    // REAL *out = malloc(n * sizeof(REAL));

    double duration;
    printf("BEFORE READOPENCL");
    cl_int err;
    cl_kernel kernel;
    size_t global[1];
    size_t local[1];
    // TODO: write our own work-unit
    char *KernelSource = readOpenCL("src/opencl/stencil.cl");
    printf("AFTER READOPENCL");
    // only works for main ofc, no argv[1] here, possibly different argument
    local[0] = atoi(argv[1]);

    float *data = NULL;                /* Original data set given to device.  */
    float *results = NULL;             /* Results returned from device.  */
   
    data = calloc(n, sizeof(float));
    data[0] = 100;
    data[n - 1] = 1000;
    results = calloc(n, sizeof(float));
    results[0] = data[0];
    results[n - 1] = data[n - 1];
   

    // probably wrong
    int count = atoi(argv[1]);
    // global[0] = count;

    // creates context and command queue, chooses device and platform
    err = initGPU();

    if(err == CL_SUCCESS) {
        // TODO: verander values
        global[0] = n;
        local[0] = n;
        // count = 1024;
        
        printf("PRE LOOP\n");
        cl_kernel kernel;
        
        for (int i = 0; i < iterations; i++) {
            kernel = setupKernel(KernelSource, "stencil", 3, 
            FloatArr, count, data, 
            FloatArr, count-1, results, 
            IntConst, count);
            // printf("POST KERNEL SETUP\n");
            runKernel(kernel, 1, global, local);

            if (i != iterations) { 
              float *temp = data;
              data = results;
              results = temp;
            } 
        }

        // cl_kernel kernel;
        // kernel = setupKernel(KernelSource, "stencil", 4, 
        //     FloatArr, count, data, 
        //     FloatArr, count-1, results, 
        //     IntConst, count,
        //     IntConst, iterations);
        // runKernel(kernel, 1, global, local);
    
        printf("Contents of results:\n");
        for (int i = 0; i < n; i ++) {
            printf("index %d: %lf \n",i,results[i]);
        }

        // printf("%lf bruh %lf", results, count);

        printKernelTime();
        printTransferTimes();

        err = clReleaseKernel (kernel);
        err = freeDevice();
  } 

    printf("%lf", duration,
            5.0 * (n - 2) * iterations / 1e9 / duration);

    return EXIT_SUCCESS;
}
