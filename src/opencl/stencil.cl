__kernel void stencil(
   __global float* input,
   __global float* output,
   const unsigned int count)
{
   
   int i = get_global_id(0);
   if (i > 0 && i < count) {
      const float a = 0.1;
      const float b = 0.2;
      const float c = 0.3;
      output[i] = a * input[i - 1] + b * input[i] + c * input[i + 1];
   }

}

