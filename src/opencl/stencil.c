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
    
    cl_int err;
    cl_kernel kernel;

    size_t global[1] = {1024};
    size_t local[1] = {256};

    err = initGPU();
    if(err != CL_SUCCESS) { return; }

    char *KernelSource = readOpenCL("src/opencl/stencil.cl");

    kernel = setupKernel(KernelSource, "stencil", 3, 
            DoubleArr, n, *in, 
            DoubleArr, n, *out, 
            IntConst, n);

    cl_mem inBuf = allocDev(sizeof(REAL) * n);
    cl_mem outBuf = allocDev(sizeof(REAL) * n);

    host2devDoubleArr(*in, inBuf, n);

    for (int t = 0; t < iterations; t++) {
        // clSetKernelArg(kernel,0,n,inBuf);
        // clSetKernelArg(kernel,1,n,outBuf);
        printf("in sha Allah werk je times %d\n", t);
        clSetKernelArg(kernel, 2, sizeof(int), &t);
        runKernel(kernel, 1, global, local);

        /* The output of this iteration is the input of the next iteration (if there is one). */
        if (t != iterations) {
            cl_mem temp = inBuf;
            inBuf = outBuf;
            outBuf = temp;
        }

        // dev2hostDoubleArr(outBuf, *out, n);
        // printf("Contents of iteration %d:\n", t);
        // for (int i = 0; i < n; i ++) {
        //     printf("index %d: %lf \n",i,*out[i]);
        // }
    }

    dev2hostDoubleArr(outBuf, *out, n);

    printf("Contents of results:\n");
    for (int i = 0; i < n; i ++) {
        printf("index %d: %lf \n",i,*out[i]);
    }

    printKernelTime();
    printTransferTimes();
    
    err = clReleaseKernel (kernel);
    err = freeDevice();
    
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
