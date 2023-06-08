// TODO: check arguments + write computation per iteration
__kernel void stencil(
   __global float* input,
   __global float* output,
   const unsigned int count)
{
   const REAL a = 0.1;
   const REAL b = 0.2;
   const REAL c = 0.3;
   int i = get_global_id(0) + 1;
   output[i] = a * input[i - 1] + b * input[i] + c * input[i + 1];
}

