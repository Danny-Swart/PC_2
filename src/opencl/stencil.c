#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <immintrin.h>

#include "../simple.h"
#include <CL/cl.h>

#define REAL double

/* You may need a different method of timing if you are not on Linux. */
#define TIME(duration, fncalls)                                        \
    do {                                                               \
        struct timeval tv1, tv2;                                       \
        gettimeofday(&tv1, NULL);                                      \
        fncalls                                                        \
        gettimeofday(&tv2, NULL);                                      \
        duration = (REAL) (tv2.tv_usec - tv1.tv_usec) / 1000000 +    \
         (REAL) (tv2.tv_sec - tv1.tv_sec);                           \
    } while (0)

const REAL a = 0.1;
const REAL b = 0.2;
const REAL c = 0.3;

void Stencil(REAL **in, REAL **out, size_t n, int iterations)
{
    cl_int err; // can be used for error checking
    cl_kernel kernel; // the actual kernel that will be initialized
    cl_mem inBuf, outBuf; // the input and output buffer

    err = initGPU();
    if (err != CL_SUCCESS) {
        fprintf(stderr, "failed to initialize OpenCL environment\n");
        return;
    }

    // allocate device memory for input and output buffers
    inBuf = allocDev(n * sizeof(REAL));
    outBuf = allocDev(n * sizeof(REAL));

    // transfer input data from host to device
    host2devDoubleArr(*in, inBuf, n);

    // create and set up the OpenCL kernel
    kernel = setupKernel(readOpenCL("src/opencl/stencil.cl"), "stencil", 3,
                         DoubleArr, n, *in,
                         DoubleArr, n, *out,
                         IntConst, n);
    if (kernel == NULL) {
        fprintf(stderr, "failed to create OpenCL kernel\n");
        return;
    }

    size_t global[1] = {n}; // set the global size to n

    // workgroup size
    size_t local[1] = {256};

    for (int t = 0; t < iterations; t++) {
        // set the iteration number as a constant kernel argument
        err = clSetKernelArg(kernel, 2, sizeof(int), &t);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "failed to set kernel argument\n");
            break;
        }

        err = launchKernel(kernel, 1, global, local);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "failed to launch OpenCL kernel\n");
            break;
        }

        // swap the input and output buffers
        cl_mem temp = inBuf;
        inBuf = outBuf;
        outBuf = temp;
    }

    // transfer the final output from device to host
    dev2hostDoubleArr(outBuf, *out, n);

    clReleaseKernel(kernel);
    freeDevice();
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        printf("Please specify 2 arguments (n, iterations).\n");
        return EXIT_FAILURE;
    }

    size_t n = atoll(argv[1]);
    int iterations = atoi(argv[2]);
    
    REAL *in = calloc(n, sizeof(REAL));
    in[0] = 100;
    in[n - 1] = 1000;
    REAL *out = calloc(n, sizeof(REAL));
    (out)[0] = (in)[0];
    (out)[n - 1] = (in)[n - 1];

    double duration;
    TIME(duration, Stencil(&in, &out, n, iterations););
    printf("%lf", duration,
            5.0 * (n - 2) * iterations / 1e9 / duration);

    printf("Contents of results:\n");
    for (int i = 0; i < n; i ++) {
        printf("index %d: %lf \n",i,out[i]);
    }

    free(in);
    free(out);

    return EXIT_SUCCESS;
}
