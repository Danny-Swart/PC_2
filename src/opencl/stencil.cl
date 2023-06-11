__kernel void stencil(
   __global float* input,
   __global float* output,
   const unsigned int count,
   const unsigned int iterations)
{
   const float a = 0.1;
   const float b = 0.2;
   const float c = 0.3;
   
   for (int j = 0; j < iterations; j++) {
      int i = get_global_id(0) + 1;
      output[i] = a * input[i - 1] + b * input[i] + c * input[i + 1];      
      if (j != iterations) { 
        float *temp = input;
        input = output;
        output = temp;
      }
   }
}

