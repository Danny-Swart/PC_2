#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <sys/time.h>
#include <immintrin.h>
#include <mpi.h>

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

void StencilMain(REAL **in, REAL **out, size_t n, int iterations, int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, rankCount;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

    (*out)[0] = (*in)[0];
    (*out)[n - 1] = (*in)[n - 1];

    if (rank == 0) {
        // StencilMaster(in, out, n, iterations, rank);
        for (int t = 1; t <= iterations; t++) {
            /* Update only the inner values. */
            for (int i = 1; i < n - 1; i++) {
                double one = (*in)[i - 1];
                double two = (*in)[i];
                double three = (*in)[i + 1];
                MPI_Send(&one, 1 , MPI_DOUBLE, t*i, 0, MPI_COMM_WORLD);
                MPI_Send(&two, 1 , MPI_DOUBLE, t*i, 0, MPI_COMM_WORLD);
                MPI_Send(&three, 1 , MPI_DOUBLE, t*i, 0, MPI_COMM_WORLD);
            }

            double *result;

            for (int i = 1; i < n - 1; i++) {
                (*out)[i] = MPI_Recv(&result[i], 1, MPI_DOUBLE, t*i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            if (t != iterations) {
                REAL *temp = *in;
                *in = *out;
                *out = temp;
            }
        }
    } else /* id != 0 */ {
        // StencilWorker();
        double one, two, three, result;
        MPI_Recv(&one, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&two, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&three, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        result = a * one + b * two + c * three;

        MPI_Send(&result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
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
    REAL *out = malloc(n * sizeof(REAL));

    double duration;

    TIME(duration, StencilMain(&in, &out, n, iterations, argc, argv););
    printf("%lf", 5.0 * (n - 2) * iterations / 1e9 / duration);

    free(in);
    free(out);

    return EXIT_SUCCESS;
}
