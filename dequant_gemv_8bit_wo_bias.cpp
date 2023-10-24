__device__ __forceinline__ float2 _16bits_to8bits_quant(myhalf2* fp16_value, half s)
{

  int8_t output[8];

  output[0] = __half2int_rn(fp16_value[0].x / s);
  output[1] = __half2int_rn(fp16_value[0].y / s);

  output[2] = __half2int_rn(fp16_value[1].x / s);
  output[3] = __half2int_rn(fp16_value[1].y / s);

  output[4] = __half2int_rn(fp16_value[2].x / s);
  output[5] = __half2int_rn(fp16_value[2].y / s);

  output[6] = __half2int_rn(fp16_value[3].x / s);
  output[7] = __half2int_rn(fp16_value[3].y / s);

  return *((float2*)&output[0]);

}


__device__ __forceinline__ float2 _4bits_to8bits_dequant(int8_t* int8_value, int8_t z)
{

  int8_t output[8];

  // negative shl
  output[0] = ((int8_value[0] >> 4) & 0x0F) - z;
  output[1] = (int8_value[0] & 0x0F) - z;

  output[2] = ((int8_value[1] >> 4) & 0x0F) - z;
  output[3] = (int8_value[1] & 0x0F) - z;
  
  output[4] = ((int8_value[2] >> 4) & 0x0F) - z;
  output[5] = (int8_value[2] & 0x0F) - z;

  output[6] = ((int8_value[3] >> 4) & 0x0F) - z;
  output[7] = (int8_value[3] & 0x0F) - z;

  return *((float2*)&output[0]);

}





__global__ void dequant_gemv_8bit_wo_bias(
                            const uint8_t *qweight, half* vec, 
                            const half *input_scales,
                            half* res, unsigned int n, unsigned int num_per_thread, 
                            const int8_t *zeros, const half *scales, 
                            const int *__restrict__ indicator, 
                            const int *__restrict__ qindicator, 
                            const int c_in, const int qc_in, 
                            const int c_out, const int gs
                            ) {

  float sum = 0.0f;
  // each thread load num_per_thread elements from global
  int tidx = threadIdx.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int block_dim_x = blockDim.x;

  int start_idx = threadIdx.x;
  myhalf2 vec_val[16];
  int8_t vec_int8[32];
  
  uint8_t qw[16];  
  int8_t w[32];        


#pragma unroll
  for (int iter = 0; iter < DIV_UP(num_per_thread, 32); iter++) {
    int j = (start_idx + iter * blockDim.x) << 5;
    if (j >= n) {break;}

    // offset computation
    int w_offset_fp16 = j;
    int w_offset_uint8 = w_offset_fp16 / 2;
    int z_offset_uint8 = w_offset_fp16 / gs / 2;

    // half z_r[2] = {z[inter_zero_idx], z[inter_zero_idx]};
    int8_t z = zeros[row * c_in / gs + j / gs];
    // half z_r[2] = {z, z};

    // load scales in fp16
    half s = scales[row * c_in / gs + j / gs];
    // half s_r[2] = {s, s};

    half input_s =  input_scales[0];

    // get vec_fp16
    *(float4*)(&vec_val[0]) = *(float4*)(&vec[j]);
    *(float4*)(&vec_val[4]) = *(float4*)(&vec[j + 8]);
    *(float4*)(&vec_val[8]) = *(float4*)(&vec[j + 16]);
    *(float4*)(&vec_val[12]) = *(float4*)(&vec[j + 24]);

    // vec_fp16 -> vec_int8  32 elem
    *(float2*)(&vec_int8[0]) = _16bits_to8bits_quant(&vec_val[0], input_s);
    *(float2*)(&vec_int8[8]) = _16bits_to8bits_quant(&vec_val[4], input_s);
    *(float2*)(&vec_int8[16]) = _16bits_to8bits_quant(&vec_val[8], input_s);
    *(float2*)(&vec_int8[24]) = _16bits_to8bits_quant(&vec_val[12], input_s);

    // get mat_int4bit  32 elem
    *(float4*)(&qw[0]) = *(float4*)(&qweight[row * qc_in + w_offset_uint8]);
    *((float2*)&w[0]) = _4bits_to8bits_dequant((int8_t*)&qw[0], z);
    *((float2*)&w[8]) = _4bits_to8bits_dequant((int8_t*)&qw[4], z);
    *((float2*)&w[16]) = _4bits_to8bits_dequant((int8_t*)&qw[8], z);
    *((float2*)&w[24]) = _4bits_to8bits_dequant((int8_t*)&qw[12], z);

     // scaling
    #pragma unroll
    for (int i = 0; i < 32; i++){
      sum += ((float)w[0] * (float)vec_int8[0]);
    }
    sum *= __half2float(s);
    sum = warpReduceSum(sum, blockDim.x);

    if (blockDim.x <= WARP_SIZE) {
        if (tidx == 0) {
          res[row] = __float2half(sum);
        }
        return;
      }

    static __shared__ half warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;
    if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE)? __half2float(warpLevelSums[threadIdx.y][laneId]) : 0.0f;
    // Final reduce using first warp
    if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
    if (tidx == 0) {
      res[row] = __float2half(sum);
    }
   
}

}
